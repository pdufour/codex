use crate::app_event::AppEvent;
use crate::app_event_sender::AppEventSender;
use hf_hub::api::sync::Api as HfHubApi;
use imageproc::drawing::draw_filled_rect_mut;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::drawing::draw_text_mut;
use imageproc::rect::Rect;
use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::CameraIndex;
use nokhwa::utils::RequestedFormat;
use nokhwa::utils::RequestedFormatType;
use od_opencv::BBox;
use od_opencv::ImageBuffer;
use od_opencv::backend_ort::ModelUltralyticsOrt;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use tokio::task;

/// Explicit ASL mapping skipping 'J' (25 classes total)
const ASL_CLASS_TO_LETTER: &[&str] = &[
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
];

/// Architecture / variant class of a local sign-language model.
#[derive(Clone, Copy, Debug)]
pub enum SignLanguageModelClass {
    AslYolo11n,
    AslYolo11s,
    AslYolo11m,
    AslYolo11l,
    AslYolo11x,
}

/// One allowed local sign-language model: identifier, class (loader), and UI label/description.
pub struct AllowedSignLanguageModel {
    pub repo_id: &'static str,
    pub class: SignLanguageModelClass,
    pub label: &'static str,
    pub description: &'static str,
}

const ALLOWED_SIGN_LANGUAGE_MODELS: &[AllowedSignLanguageModel] = &[
    AllowedSignLanguageModel {
        repo_id: "asl-yolo-11n",
        class: SignLanguageModelClass::AslYolo11n,
        label: "ASL YOLO11n (smallest)",
        description: "Fastest, smallest generic YOLO11n ONNX model.",
    },
    AllowedSignLanguageModel {
        repo_id: "asl-yolo-11s",
        class: SignLanguageModelClass::AslYolo11s,
        label: "ASL YOLO11s (small)",
        description: "Small generic YOLO11s-style ONNX model (mapped to YOLO11n).",
    },
    AllowedSignLanguageModel {
        repo_id: "asl-yolo-11m",
        class: SignLanguageModelClass::AslYolo11m,
        label: "ASL YOLO11m (medium)",
        description: "Balanced medium-style option (currently mapped to YOLO11n ONNX).",
    },
    AllowedSignLanguageModel {
        repo_id: "asl-yolo-11l",
        class: SignLanguageModelClass::AslYolo11l,
        label: "ASL YOLO11l (large)",
        description: "Larger-style option (currently mapped to YOLO11n ONNX).",
    },
    AllowedSignLanguageModel {
        repo_id: "asl-yolo-11x",
        class: SignLanguageModelClass::AslYolo11x,
        label: "ASL YOLO11x (xlarge)",
        description: "Xlarge-style option (currently mapped to YOLO11n ONNX).",
    },
];

fn allowed_sign_language_repo_ids() -> impl Iterator<Item = &'static str> {
    ALLOWED_SIGN_LANGUAGE_MODELS.iter().map(|m| m.repo_id)
}

pub(crate) fn allowed_sign_language_models() -> String {
    allowed_sign_language_repo_ids()
        .collect::<Vec<_>>()
        .join(", ")
}

pub(crate) fn sign_language_model_picker_options() -> Vec<(&'static str, &'static str, &'static str)>
{
    ALLOWED_SIGN_LANGUAGE_MODELS
        .iter()
        .map(|m| (m.label, m.repo_id, m.description))
        .collect()
}

static SELECTED_SIGN_LANGUAGE_MODEL: OnceLock<Mutex<Option<String>>> = OnceLock::new();

pub(crate) fn selected_sign_language_model() -> Option<String> {
    let guard = SELECTED_SIGN_LANGUAGE_MODEL
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("selected sign-language model lock poisoned");
    guard.clone()
}

pub(crate) fn try_set_selected_sign_language_model(model: impl Into<String>) -> Result<(), String> {
    let trimmed = model.into().trim().to_string();
    if !allowed_sign_language_repo_ids().any(|id| id == trimmed) {
        let allowed = allowed_sign_language_repo_ids()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "Unsupported sign-language model '{trimmed}'. Allowed values: {allowed}."
        ));
    }
    let mut guard = SELECTED_SIGN_LANGUAGE_MODEL
        .get_or_init(|| Mutex::new(None))
        .lock()
        .expect("selected sign-language model lock poisoned");
    *guard = Some(trimmed);
    Ok(())
}

fn resolve_sign_language_model_selection() -> Result<String, String> {
    selected_sign_language_model()
        .ok_or_else(|| "No sign-language model selected; use /signmodel to pick one.".to_string())
}

pub struct RecordedVideo {
    pub data: Arc<Vec<u8>>,
}

pub trait SignLanguageModel: Send {
    fn transcribe_video(&self, video: &RecordedVideo) -> Result<String, String>;
}

struct AslYoloOrtModel {
    model: Mutex<ModelUltralyticsOrt>,
}

impl AslYoloOrtModel {
    fn new(model_path: &str) -> Result<Self, String> {
        let _ = ort::init();
        let net_size = (640, 640);
        let class_filters: Vec<usize> = Vec::new();
        tracing::info!(model_path, "sign-language: loading ONNX model");
        let model = ModelUltralyticsOrt::new_from_file(model_path, net_size, class_filters)
            .map_err(|e| format!("failed to load ASL YOLO ONNX model from {model_path}: {e}"))?;
        tracing::info!(model_path, "sign-language: model loaded");
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn detect_frame_rgb(
        &self,
        img: &image::RgbImage,
    ) -> Result<(Option<(usize, f32)>, Vec<(BBox, usize, f32)>), String> {
        let img_buf = ImageBuffer::from_dynamic_image(image::DynamicImage::ImageRgb8(img.clone()));
        let mut guard = self
            .model
            .lock()
            .map_err(|_| "ASL model lock poisoned".to_string())?;

        let (bboxes, class_ids, confidences) = guard
            .forward(&img_buf, 0.01, 0.45)
            .map_err(|e| format!("inference failed: {e}"))?;

        tracing::info!(
            detections = class_ids.len(),
            "sign-language: frame inference completed"
        );

        if class_ids.is_empty() {
            tracing::warn!(
                "sign-language: absolutely no detections even at 0.01 confidence. (Check color space!)"
            );
            return Ok((None, Vec::new()));
        }

        let len = bboxes.len().min(class_ids.len()).min(confidences.len());
        let mut detections: Vec<(BBox, usize, f32)> = Vec::with_capacity(len);
        for i in 0..len {
            let bbox = bboxes[i];
            let class_id = class_ids[i];
            let confidence = confidences[i];
            tracing::info!(
                index = i,
                class_id = class_id,
                confidence = confidence,
                x = bbox.x,
                y = bbox.y,
                width = bbox.width,
                height = bbox.height,
                "sign-language: detected class_id={class_id} confidence={confidence}"
            );
            detections.push((bbox, class_id, confidence));
        }

        let best = detections
            .iter()
            .map(|(_, class_id, conf)| (*class_id, *conf))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok((best, detections))
    }
}

fn try_load_overlay_font() -> Option<ab_glyph::FontArc> {
    // Prefer common system fonts instead of bundling assets.
    let candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ];
    for path in candidates {
        if let Ok(bytes) = std::fs::read(path)
            && let Ok(font) = ab_glyph::FontArc::try_from_vec(bytes)
        {
            return Some(font);
        }
    }
    None
}

fn overlay_font() -> Option<&'static ab_glyph::FontArc> {
    static FONT: OnceLock<Option<ab_glyph::FontArc>> = OnceLock::new();
    FONT.get_or_init(try_load_overlay_font).as_ref()
}

fn draw_detection_overlay(img: &mut image::RgbImage, detections: &[(BBox, usize, f32)]) {
    let red = image::Rgb([255, 0, 0]);
    let white = image::Rgb([255, 255, 255]);
    let text_scale = 18.0f32;
    let (img_w, img_h) = img.dimensions();
    let font = overlay_font();
    for (bbox, class_id, confidence) in detections {
        let x0 = bbox.x.max(0) as u32;
        let y0 = bbox.y.max(0) as u32;
        let x1 = (bbox.x + bbox.width - 1).max(0) as u32;
        let y1 = (bbox.y + bbox.height - 1).max(0) as u32;
        if x0 >= img_w || y0 >= img_h {
            continue;
        }
        let x1 = x1.min(img_w.saturating_sub(1));
        let y1 = y1.min(img_h.saturating_sub(1));
        if x0 > x1 || y0 > y1 {
            continue;
        }

        let rect = Rect::at(x0 as i32, y0 as i32).of_size(
            (x1.saturating_sub(x0)).max(1) + 1,
            (y1.saturating_sub(y0)).max(1) + 1,
        );
        draw_hollow_rect_mut(img, rect, red);

        let letter = if *class_id < ASL_CLASS_TO_LETTER.len() {
            ASL_CLASS_TO_LETTER[*class_id]
        } else {
            "?"
        };
        let label = format!("{letter} {:.2}", confidence);
        let label_w = ((label.len() as u32) * 10).max(36);
        let label_h = 22u32;
        let label_x = x0;
        let label_y = y0.saturating_sub(label_h);
        let bg_rect = Rect::at(label_x as i32, label_y as i32).of_size(
            label_w.min(img_w.saturating_sub(label_x)),
            label_h.min(img_h.saturating_sub(label_y)),
        );
        draw_filled_rect_mut(img, bg_rect, red);
        if let Some(font) = font {
            draw_text_mut(
                img,
                white,
                (label_x + 3) as i32,
                (label_y + 2) as i32,
                text_scale,
                font,
                &label,
            );
        }
    }
}

impl SignLanguageModel for AslYoloOrtModel {
    fn transcribe_video(&self, _video: &RecordedVideo) -> Result<String, String> {
        let index = CameraIndex::Index(0);
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        tracing::info!("sign-language: opening camera");
        let mut cam =
            Camera::new(index, requested).map_err(|e| format!("failed to open camera: {e}"))?;
        #[cfg(target_os = "macos")]
        tracing::info!("sign-language: opening camera stream");
        #[cfg(target_os = "macos")]
        cam.open_stream()
            .map_err(|e| format!("failed to start camera stream: {e}"))?;
        tracing::info!("sign-language: camera ready");

        let debug_frames_dir: Option<PathBuf> = env::var_os("CODEX_SIGN_DEBUG_FRAMES")
            .filter(|v| !v.is_empty())
            .map(|v| {
                let s = v.to_string_lossy();
                if s.eq_ignore_ascii_case("1")
                    || s.eq_ignore_ascii_case("true")
                    || s.eq_ignore_ascii_case("yes")
                {
                    PathBuf::from("/tmp/codex-debug-frames")
                } else {
                    PathBuf::from(v)
                }
            });

        if let Some(ref dir) = debug_frames_dir {
            if let Err(e) = std::fs::create_dir_all(dir) {
                tracing::warn!(path = %dir.display(), error = %e, "sign-language: failed to create debug frames dir");
            } else {
                tracing::info!(path = %dir.display(), "sign-language: writing debug frames");
            }
        }

        let mut counts: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
        let mut confidence_sums: std::collections::HashMap<usize, f32> =
            std::collections::HashMap::new();
        let capture_duration = Duration::from_secs(5);
        let deadline = Instant::now() + capture_duration;
        let mut frame_index = 0u32;
        tracing::info!(
            capture_seconds = capture_duration.as_secs(),
            "sign-language: capture started"
        );

        while Instant::now() < deadline {
            let current_frame_index = frame_index;
            let frame = cam
                .frame()
                .map_err(|e| format!("failed to capture frame: {e}"))?;
            let decoded = frame
                .decode_image::<RgbFormat>()
                .map_err(|e| format!("failed to decode frame: {e}"))?;

            let (prediction, detections) = self.detect_frame_rgb(&decoded)?;

            if let Some(ref dir) = debug_frames_dir {
                let path = dir.join(format!("frame_{frame_index:03}.png"));
                let mut overlay = decoded.clone();
                draw_detection_overlay(&mut overlay, &detections);
                if let Err(e) = image::DynamicImage::ImageRgb8(overlay).save(&path) {
                    tracing::warn!(path = %path.display(), error = %e, "sign-language: failed to write debug frame");
                }
            }
            frame_index += 1;

            if let Some((class_id, conf)) = prediction {
                *counts.entry(class_id).or_insert(0) += 1;
                *confidence_sums.entry(class_id).or_insert(0.0) += conf;
                let letter = if class_id < ASL_CLASS_TO_LETTER.len() {
                    ASL_CLASS_TO_LETTER[class_id]
                } else {
                    "?"
                };
                tracing::info!(
                    frame_index = current_frame_index,
                    class_id = class_id,
                    letter = %letter,
                    confidence = conf,
                    "sign-language: frame prediction"
                );
            } else {
                tracing::info!(
                    frame_index = current_frame_index,
                    "sign-language: frame prediction none"
                );
            }
        }
        tracing::info!(
            frames_captured = frame_index,
            "sign-language: capture finished"
        );
        tracing::info!(vote_counts = ?counts, "sign-language: class vote summary");
        tracing::info!(
            confidence_sums = ?confidence_sums,
            "sign-language: class confidence summary"
        );

        let best_class = confidence_sums
            .iter()
            .max_by(|(class_a, sum_a), (class_b, sum_b)| {
                sum_a
                    .partial_cmp(sum_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        counts
                            .get(class_a)
                            .unwrap_or(&0)
                            .cmp(counts.get(class_b).unwrap_or(&0))
                    })
                    .then_with(|| class_b.cmp(class_a))
            })
            .map(|(class_id, _)| *class_id);

        if let Some(class_id) = best_class {
            // Safely map to the correct ASL letter array
            let letter = if class_id < ASL_CLASS_TO_LETTER.len() {
                ASL_CLASS_TO_LETTER[class_id]
            } else {
                "?"
            };

            tracing::info!(class_id, letter = %letter, "sign-language: final prediction");
            Ok(letter.to_string())
        } else {
            tracing::warn!("sign-language: no winning class after capture window");
            Ok("(no sign detected)".to_string())
        }
    }
}

fn load_model_for_class(
    class: SignLanguageModelClass,
) -> Result<Box<dyn SignLanguageModel>, String> {
    tracing::info!(?class, "sign-language: resolving model class");
    let filename = match class {
        SignLanguageModelClass::AslYolo11n => "yolo11n.onnx",
        SignLanguageModelClass::AslYolo11s => "yolo11s.onnx",
        SignLanguageModelClass::AslYolo11m => "yolo11m.onnx",
        SignLanguageModelClass::AslYolo11l => "yolo11l.onnx",
        SignLanguageModelClass::AslYolo11x => "yolo11x.onnx",
    };

    let model_path = ensure_sign_model_downloaded(filename)?;
    tracing::info!(filename, path = %model_path.display(), "sign-language: model ready on disk");
    let model = AslYoloOrtModel::new(
        model_path
            .to_str()
            .ok_or_else(|| "non-UTF8 sign model path".to_string())?,
    )?;
    Ok(Box::new(model))
}

fn get_class_for_repo(repo_id: &str) -> Option<SignLanguageModelClass> {
    ALLOWED_SIGN_LANGUAGE_MODELS
        .iter()
        .find(|m| m.repo_id == repo_id)
        .map(|m| m.class)
}

fn resolve_sign_language_model() -> Result<Box<dyn SignLanguageModel>, String> {
    let model_id = resolve_sign_language_model_selection()?;
    tracing::info!(model_id, "sign-language: selected model id");
    let class = get_class_for_repo(&model_id)
        .ok_or_else(|| format!("unknown sign-language model repo: {model_id}"))?;
    load_model_for_class(class)
}

pub(crate) async fn prefetch_selected_sign_language_model(model_id: &str) -> Result<(), String> {
    let class = get_class_for_repo(model_id)
        .ok_or_else(|| format!("unknown sign-language model repo: {model_id}"))?;

    let filename = match class {
        SignLanguageModelClass::AslYolo11n => "yolo11n.onnx",
        SignLanguageModelClass::AslYolo11s => "yolo11s.onnx",
        SignLanguageModelClass::AslYolo11m => "yolo11m.onnx",
        SignLanguageModelClass::AslYolo11l => "yolo11l.onnx",
        SignLanguageModelClass::AslYolo11x => "yolo11x.onnx",
    };

    task::spawn_blocking(move || ensure_sign_model_downloaded(filename))
        .await
        .map_err(|e| format!("sign-language prefetch task failed: {e}"))??;
    Ok(())
}

pub fn transcribe_video_async(id: String, video: RecordedVideo, tx: AppEventSender) {
    thread::spawn(move || {
        let res = resolve_sign_language_model().and_then(|model| model.transcribe_video(&video));
        match res {
            Ok(text) => {
                tx.send(AppEvent::TranscriptionComplete { id, text });
            }
            Err(error) => {
                tx.send(AppEvent::TranscriptionFailed { id, error });
            }
        }
    });
}

pub fn capture_sign_letter() -> Result<String, String> {
    let model = resolve_sign_language_model()?;
    let video = RecordedVideo {
        data: Arc::new(Vec::new()),
    };
    model.transcribe_video(&video)
}

fn ensure_sign_model_downloaded(filename: &str) -> Result<PathBuf, String> {
    let repo_id = "pdufour/asl-yolo-models-onnx";
    let api = HfHubApi::new().map_err(|e| format!("failed to init hf-hub API: {e}"))?;
    let repo = api.model(repo_id.to_string());
    repo.get(filename)
        .map_err(|e| format!("failed to download sign model {filename} from {repo_id}: {e}"))
}
