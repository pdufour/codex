use crate::app_event::AppEvent;
use crate::app_event_sender::AppEventSender;
use hf_hub::api::sync::Api as HfHubApi;
use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::CameraIndex;
use nokhwa::utils::RequestedFormat;
use nokhwa::utils::RequestedFormatType;
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

/// Architecture / variant class of a local sign-language model.
#[derive(Clone, Copy)]
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
    /// Display name in the model picker (e.g. "SignLang Basic").
    pub label: &'static str,
    /// Short description for the picker.
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

/// Options for the sign-language model picker: (display label, model id, description).
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

/// Simple in-process representation of a video recording.
///
/// This is intentionally minimal for now; callers are expected to construct it from whatever
/// capture mechanism they use.
pub struct RecordedVideo {
    /// Opaque bytes for the recorded video or sequence of frames.
    pub data: Arc<Vec<u8>>,
}

/// Trait implemented by local sign-language models.
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
        let model = ModelUltralyticsOrt::new_from_file(model_path, net_size, class_filters)
            .map_err(|e| format!("failed to load ASL YOLO ONNX model from {model_path}: {e}"))?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    fn detect_frame_rgb(&self, img: &image::RgbImage) -> Result<Option<(usize, f32)>, String> {
        let img_buf = ImageBuffer::from_dynamic_image(image::DynamicImage::ImageRgb8(img.clone()));
        let mut guard = self
            .model
            .lock()
            .map_err(|_| "ASL model lock poisoned".to_string())?;
        // Match ASL-Detector-YOLO Space: conf=0.45, iou=0.7 (NMS).
        let (_bboxes, class_ids, confidences) = guard
            .forward(&img_buf, 0.45, 0.7)
            .map_err(|e| format!("inference failed: {e}"))?;

        let best = class_ids
            .into_iter()
            .zip(confidences.into_iter())
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(best)
    }
}

impl SignLanguageModel for AslYoloOrtModel {
    fn transcribe_video(&self, _video: &RecordedVideo) -> Result<String, String> {
        let index = CameraIndex::Index(0);
        let requested =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        let mut cam =
            Camera::new(index, requested).map_err(|e| format!("failed to open camera: {e}"))?;
        #[cfg(target_os = "macos")]
        cam.open_stream()
            .map_err(|e| format!("failed to start camera stream: {e}"))?;

        let debug_frames_dir: Option<PathBuf> = env::var_os("CODEX_SIGN_DEBUG_FRAMES")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);
        if let Some(ref dir) = debug_frames_dir {
            let _ = std::fs::create_dir_all(dir);
        }

        let mut counts: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
        let capture_duration = Duration::from_secs(10);
        let deadline = Instant::now() + capture_duration;
        let mut frame_index = 0u32;
        while Instant::now() < deadline {
            let frame = cam
                .frame()
                .map_err(|e| format!("failed to capture frame: {e}"))?;
            let decoded = frame
                .decode_image::<RgbFormat>()
                .map_err(|e| format!("failed to decode frame: {e}"))?;

            if let Some(ref dir) = debug_frames_dir {
                let path = dir.join(format!("frame_{frame_index:03}.png"));
                let _ = image::DynamicImage::ImageRgb8(decoded.clone()).save(&path);
            }
            frame_index += 1;

            if let Some((class_id, _conf)) = self.detect_frame_rgb(&decoded)? {
                *counts.entry(class_id).or_insert(0) += 1;
            }
        }

        let best_class = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(class_id, _)| class_id);

        if let Some(class_id) = best_class {
            let ch = (b'A' + (class_id as u8)) as char;
            Ok(ch.to_string())
        } else {
            Ok("(no sign detected)".to_string())
        }
    }
}

fn load_model_for_class(
    class: SignLanguageModelClass,
) -> Result<Box<dyn SignLanguageModel>, String> {
    let filename = match class {
        SignLanguageModelClass::AslYolo11n => "yolo11n.onnx",
        SignLanguageModelClass::AslYolo11s => "yolo11s.onnx",
        SignLanguageModelClass::AslYolo11m => "yolo11m.onnx",
        SignLanguageModelClass::AslYolo11l => "yolo11l.onnx",
        SignLanguageModelClass::AslYolo11x => "yolo11x.onnx",
    };

    let model_path = ensure_sign_model_downloaded(filename)?;
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
    // Download the chosen ASL YOLO ONNX model from the per-user repo.
    let repo_id = "pdufour/asl-yolo-models-onnx";
    let api = HfHubApi::new().map_err(|e| format!("failed to init hf-hub API: {e}"))?;
    let repo = api.model(repo_id.to_string());
    repo.get(filename)
        .map_err(|e| format!("failed to download sign model {filename} from {repo_id}: {e}"))
}
