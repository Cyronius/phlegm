//! Chat image input (SERVER-IMAGE-INPUT): OpenAI `image_url` content parts become the bridge's
//! placeholder text, each decoded image is resized the Qwen2-VL way (`smart_resize`) to RGB8, and
//! after tokenizing each placeholder's image token is expanded to the image's run of tokens.
//! Every number comes from the bridge's `init` reply (`vision`); nothing here is model-specific.

use serde_json::Value;
use sha2::{Digest, Sha256};

/// At most this many images in one request.
pub const MAX_IMAGES: usize = 8;
/// Largest decoded image accepted, in pixels (before resizing).
pub const MAX_SOURCE_PIXELS: u64 = 50_000_000;

/// The bridge's `vision` field of its `init` reply.
#[derive(Clone, Debug, PartialEq)]
pub struct VisionConfig {
    pub image_token_id: u32,
    pub patch: u32,
    pub merge: u32,
    pub min_pixels: u64,
    pub max_pixels: u64,
    /// Normalization the bridge applies to the RGB8 pixels it receives (informational here).
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// Text spliced into a message for each image; it must tokenize to one `image_token_id`.
    pub placeholder: String,
}

impl VisionConfig {
    /// `None` for a missing or null field (the model takes no images).
    pub fn from_init(v: Option<&Value>) -> Result<Option<VisionConfig>, String> {
        let Some(v) = v.filter(|v| !v.is_null()) else { return Ok(None) };
        let bad = || "NPU engine sent an invalid vision description".to_string();
        let uint = |k: &str| v.get(k).and_then(Value::as_u64).ok_or_else(bad);
        let triple = |k: &str| -> Result<[f32; 3], String> {
            let a = v.get(k).and_then(Value::as_array).filter(|a| a.len() == 3).ok_or_else(bad)?;
            let f = |i: usize| a[i].as_f64().map(|x| x as f32).ok_or_else(bad);
            Ok([f(0)?, f(1)?, f(2)?])
        };
        let cfg = VisionConfig {
            image_token_id: u32::try_from(uint("image_token_id")?).map_err(|_| bad())?,
            patch: u32::try_from(uint("patch")?).map_err(|_| bad())?,
            merge: u32::try_from(uint("merge")?).map_err(|_| bad())?,
            min_pixels: uint("min_pixels")?,
            max_pixels: uint("max_pixels")?,
            mean: triple("mean")?,
            std: triple("std")?,
            placeholder: v.get("placeholder").and_then(Value::as_str).ok_or_else(bad)?.to_owned(),
        };
        if cfg.patch == 0 || cfg.merge == 0 || cfg.min_pixels > cfg.max_pixels || cfg.placeholder.is_empty() {
            return Err(bad());
        }
        Ok(Some(cfg))
    }

    /// Resized sides are multiples of this (patch * merge).
    pub fn factor(&self) -> u32 {
        self.patch * self.merge
    }

    /// The image's tokens in the prompt: one per merged patch group.
    pub fn run_len(&self, width: u32, height: u32) -> usize {
        ((width / self.patch) as usize * (height / self.patch) as usize) / (self.merge * self.merge) as usize
    }
}

/// One prompt image, resized and ready for the bridge: RGB8, row-major, `width * height * 3`
/// bytes. Its position in the prompt is implied: the images' runs appear in order.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptImage {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    /// SHA-256 of `rgb`: prompt reuse keeps an image's rows only for the same pixels.
    pub sha256: [u8; 32],
}

impl PromptImage {
    pub fn new(width: u32, height: u32, rgb: Vec<u8>) -> PromptImage {
        let sha256 = Sha256::digest(&rgb).into();
        PromptImage { width, height, rgb, sha256 }
    }
}

/// Qwen2-VL `smart_resize` with the bridge's numbers (mirrors the C++ helper; rounding is half
/// away from zero). Returns (width, height).
pub fn smart_resize(width: u32, height: u32, cfg: &VisionConfig) -> Result<(u32, u32), String> {
    if width == 0 || height == 0 {
        return Err("image has no pixels".into());
    }
    let (w, h, f) = (width as f64, height as f64, cfg.factor() as f64);
    if w.max(h) / w.min(h) > 200.0 {
        return Err(format!("image aspect ratio {width}x{height} exceeds 200:1"));
    }
    let round_to = |v: f64| f.max((v / f).round() * f);
    let (mut hb, mut wb) = (round_to(h), round_to(w));
    let (min, max) = (cfg.min_pixels as f64, cfg.max_pixels as f64);
    if hb * wb > max {
        let beta = (h * w / max).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if hb * wb < min {
        let beta = (min / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((wb as u32, hb as u32))
}

fn percent_decode(s: &str) -> Result<String, String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let hex = s.get(i + 1..i + 3).and_then(|h| u8::from_str_radix(h, 16).ok());
            out.push(hex.ok_or("invalid percent escape in file URL")?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| "file URL is not UTF-8".into())
}

/// The bytes behind an image URL: a `data:` URL (base64), or, with `allow_file` (the local CLI),
/// an absolute `file://` path. No network fetching.
fn source_bytes(url: &str, allow_file: bool) -> Result<Vec<u8>, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (header, data) = rest.split_once(',').ok_or("invalid data URL")?;
        if !header.split(';').any(|p| p.eq_ignore_ascii_case("base64")) {
            return Err("image data URLs must be base64".into());
        }
        let data: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
        return base64::decode(data).map_err(|e| format!("invalid base64 image data: {e}"));
    }
    if let Some(path) = url.strip_prefix("file://").filter(|_| allow_file) {
        let path = percent_decode(path)?;
        if !std::path::Path::new(&path).is_absolute() {
            return Err("file:// image URLs must name an absolute path".into());
        }
        return std::fs::read(&path).map_err(|e| format!("reading image {path}: {e}"));
    }
    let lower = url.to_ascii_lowercase();
    if lower.starts_with("http://") || lower.starts_with("https://") {
        return Err("image URLs must be data: URLs; phlegm does not fetch http(s) images".into());
    }
    Err("image URLs must be data: URLs".into())
}

/// Decode (png, jpeg, webp, gif's first frame), composite alpha over white, resize to
/// `smart_resize` with Catmull-Rom.
pub fn prepare(bytes: &[u8], cfg: &VisionConfig) -> Result<PromptImage, String> {
    use std::io::Cursor;
    let reader = || img::ImageReader::new(Cursor::new(bytes)).with_guessed_format()
        .map_err(|e| format!("reading image: {e}"));
    let (w, h) = reader()?.into_dimensions().map_err(|e| format!("decoding image: {e}"))?;
    if w as u64 * h as u64 > MAX_SOURCE_PIXELS {
        return Err(format!("image is {w}x{h}; at most {MAX_SOURCE_PIXELS} pixels"));
    }
    let (wb, hb) = smart_resize(w, h, cfg)?;
    let decoded = reader()?.decode().map_err(|e| format!("decoding image: {e}"))?;
    let rgb = if decoded.color().has_alpha() {
        let rgba = decoded.to_rgba8();
        img::RgbImage::from_fn(w, h, |x, y| {
            let p = rgba.get_pixel(x, y).0;
            let a = p[3] as u32;
            img::Rgb([0, 1, 2].map(|c| ((p[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8))
        })
    } else {
        decoded.to_rgb8()
    };
    drop(decoded);
    let resized = if (wb, hb) == (w, h) {
        rgb
    } else {
        img::imageops::resize(&rgb, wb, hb, img::imageops::FilterType::CatmullRom)
    };
    Ok(PromptImage::new(wb, hb, resized.into_raw()))
}

/// A chat message's `content` as one string: text parts in order, each `image_url` part replaced
/// by the bridge's placeholder, its image prepared and pushed onto `images`.
pub fn message_content(
    content: Option<&Value>,
    cfg: Option<&VisionConfig>,
    allow_file: bool,
    images: &mut Vec<PromptImage>,
) -> Result<String, String> {
    let parts = match content {
        Some(Value::String(s)) => return Ok(s.clone()),
        Some(Value::Array(parts)) => parts,
        _ => return Ok(String::new()),
    };
    let mut out = String::new();
    for p in parts {
        let image = p.get("image_url").filter(|_| p.get("type").and_then(Value::as_str).is_none_or(|t| t == "image_url"));
        let Some(image) = image else {
            if let Some(t) = p.get("text").and_then(Value::as_str) {
                out.push_str(t);
            }
            continue;
        };
        let cfg = cfg.ok_or("this model does not accept images")?;
        if images.len() >= MAX_IMAGES {
            return Err(format!("at most {MAX_IMAGES} images per request"));
        }
        let url = image.as_str().or_else(|| image.get("url").and_then(Value::as_str))
            .ok_or("image_url must be a string or an object with a url")?;
        images.push(prepare(&source_bytes(url, allow_file)?, cfg)?);
        out.push_str(&cfg.placeholder);
    }
    Ok(out)
}

/// A tokenized prompt with each image's single token expanded to its run.
pub struct Expansion {
    pub ids: Vec<u32>,
    /// (position of the single token in the unexpanded prompt, run length), in order.
    singles: Vec<(usize, usize)>,
}

impl Expansion {
    /// An unexpanded prompt position in the expanded prompt.
    pub fn map(&self, pos: usize) -> usize {
        pos + self.singles.iter().filter(|&&(at, _)| at < pos).map(|&(_, n)| n - 1).sum::<usize>()
    }
}

/// Expand each `image_token_id` in `ids` to its image's run. The count of image tokens must equal
/// the count of images (a user typing the placeholder text must not shift the images).
pub fn expand(ids: &[u32], images: &[PromptImage], cfg: &VisionConfig) -> Result<Expansion, String> {
    let found = ids.iter().filter(|&&id| id == cfg.image_token_id).count();
    if found != images.len() {
        return Err(format!("prompt has {found} image tokens for {} images", images.len()));
    }
    let mut out = Vec::with_capacity(ids.len());
    let mut singles = Vec::new();
    let mut next = images.iter();
    for (at, &id) in ids.iter().enumerate() {
        if id == cfg.image_token_id {
            let image = next.next().unwrap();
            let n = cfg.run_len(image.width, image.height);
            if n == 0 {
                return Err("image too small for the vision tower".into());
            }
            singles.push((at, n));
            out.extend(std::iter::repeat(id).take(n));
        } else {
            out.push(id);
        }
    }
    Ok(Expansion { ids: out, singles })
}

/// An image's run in an expanded prompt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImageRun {
    pub start: usize,
    pub len: usize,
    pub sha256: [u8; 32],
    /// Index into the request's images.
    pub image: usize,
}

impl ImageRun {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
    /// Whether `pos` falls strictly inside the run (after its first token).
    pub fn splits(&self, pos: usize) -> bool {
        pos > self.start && pos < self.end()
    }
}

/// The images' runs in an expanded prompt, in order. Each image's sides must be multiples of
/// `factor`, and the prompt's image tokens must be exactly the images' runs.
pub fn image_runs(prompt: &[u32], images: &[PromptImage], cfg: Option<&VisionConfig>) -> Result<Vec<ImageRun>, String> {
    let Some(cfg) = cfg else {
        return if images.is_empty() { Ok(Vec::new()) } else { Err("this model does not accept images".into()) };
    };
    let f = cfg.factor();
    let mut runs = Vec::with_capacity(images.len());
    let mut at = 0;
    let mut next = images.iter().enumerate();
    while at < prompt.len() {
        if prompt[at] != cfg.image_token_id {
            at += 1;
            continue;
        }
        let (index, image) = next.next().ok_or("prompt has more image token runs than images")?;
        if image.width == 0 || image.height == 0 || image.width % f != 0 || image.height % f != 0
            || image.rgb.len() != image.width as usize * image.height as usize * 3
        {
            return Err("prompt image is not resized for the vision tower".into());
        }
        let len = cfg.run_len(image.width, image.height);
        if prompt.get(at..at + len).is_none_or(|s| s.iter().any(|&id| id != cfg.image_token_id)) {
            return Err("prompt image token run does not match its image".into());
        }
        runs.push(ImageRun { start: at, len, sha256: image.sha256, image: index });
        at += len;
    }
    if next.next().is_some() {
        return Err("prompt has fewer image token runs than images".into());
    }
    Ok(runs)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn fake_config(image_token_id: u32, placeholder: &str) -> VisionConfig {
        VisionConfig::from_init(Some(&json!({"image_token_id": image_token_id, "patch": 16, "merge": 2,
            "min_pixels": 65536, "max_pixels": 1003520, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
            "placeholder": placeholder}))).unwrap().unwrap()
    }

    pub(crate) fn png_data_url(w: u32, h: u32) -> String {
        let image = img::RgbaImage::from_fn(w, h, |x, y| img::Rgba([x as u8, y as u8, 7, if x == 0 { 0 } else { 255 }]));
        let mut bytes = Vec::new();
        img::DynamicImage::ImageRgba8(image)
            .write_to(&mut std::io::Cursor::new(&mut bytes), img::ImageFormat::Png).unwrap();
        format!("data:image/png;base64,{}", base64::encode(bytes))
    }

    #[test]
    fn smart_resize_follows_qwen2_vl() {
        let cfg = fake_config(9, "<img>");
        assert_eq!(cfg.factor(), 32);
        assert_eq!(smart_resize(640, 480, &cfg), Ok((640, 480)));
        assert_eq!(smart_resize(4000, 3000, &cfg), Ok((1152, 864)));
        assert_eq!(smart_resize(50, 50, &cfg), Ok((256, 256)));
        assert_eq!(smart_resize(1920, 1080, &cfg), Ok((1312, 736)));
        assert_eq!(smart_resize(300, 2, &cfg), Ok((3136, 32)));
        assert!(smart_resize(1, 300, &cfg).is_err());
        assert!(smart_resize(0, 30, &cfg).is_err());
        assert_eq!(cfg.run_len(640, 480), 300);
        assert_eq!(VisionConfig::from_init(Some(&Value::Null)), Ok(None));
        assert!(VisionConfig::from_init(Some(&json!({"patch": 16}))).is_err());
    }

    #[test]
    fn content_parts_become_placeholders_and_images() {
        let cfg = fake_config(9, "<img>");
        let content = json!([{"type": "text", "text": "a "},
            {"type": "image_url", "image_url": {"url": png_data_url(64, 48)}},
            {"type": "text", "text": " b"}, {"type": "image_url", "image_url": png_data_url(40, 40)}]);
        let mut images = Vec::new();
        let text = message_content(Some(&content), Some(&cfg), false, &mut images).unwrap();
        assert_eq!(text, "a <img> b<img>");
        // 64x48 rounds to 64x64, below min_pixels -> 320x224; 40x40 -> 256x256.
        assert_eq!(images.iter().map(|i| (i.width, i.height)).collect::<Vec<_>>(), [(320, 224), (256, 256)]);
        assert_eq!(images[0].rgb.len(), 320 * 224 * 3);
        assert_eq!(images[0].sha256, <[u8; 32]>::from(Sha256::digest(&images[0].rgb)));
        // The transparent first column is composited over white.
        assert_eq!(&images[0].rgb[..3], &[255, 255, 255]);
        let mut none = Vec::new();
        for (url, allow) in [("https://example.com/a.png", false), ("file:///tmp/a.png", false),
                             ("data:image/png;base64,!!", false), ("data:text/plain,hi", false)] {
            let c = json!([{"type": "image_url", "image_url": {"url": url}}]);
            assert!(message_content(Some(&c), Some(&cfg), allow, &mut none).is_err(), "{url}");
        }
        let c = json!([{"type": "image_url", "image_url": {"url": png_data_url(32, 32)}}]);
        assert_eq!(message_content(Some(&c), None, false, &mut none), Err("this model does not accept images".into()));
        let many = json!((0..=MAX_IMAGES).map(|_| json!({"type": "image_url", "image_url": png_data_url(32, 32)})).collect::<Vec<_>>());
        assert!(message_content(Some(&many), Some(&cfg), false, &mut none).unwrap_err().contains("at most"));
    }

    #[test]
    fn expansion_and_runs() {
        let cfg = fake_config(9, "<img>");
        let a = PromptImage::new(64, 32, vec![1; 64 * 32 * 3]); // 4x2 patches -> 2 tokens
        let b = PromptImage::new(64, 64, vec![2; 64 * 64 * 3]); // 4 tokens
        let e = expand(&[1, 9, 2, 9, 3], &[a.clone(), b.clone()], &cfg).unwrap();
        assert_eq!(e.ids, [1, 9, 9, 2, 9, 9, 9, 9, 3]);
        assert_eq!((e.map(0), e.map(1), e.map(2), e.map(3), e.map(5)), (0, 1, 3, 4, 9));
        assert!(expand(&[1, 9, 9], &[a.clone()], &cfg).is_err());
        let runs = image_runs(&e.ids, &[a.clone(), b.clone()], Some(&cfg)).unwrap();
        assert_eq!(runs.iter().map(|r| (r.start, r.len, r.image)).collect::<Vec<_>>(), [(1, 2, 0), (4, 4, 1)]);
        assert!(runs[1].splits(5) && !runs[1].splits(4) && !runs[1].splits(8));
        assert!(image_runs(&e.ids, &[a.clone()], Some(&cfg)).is_err());
        assert!(image_runs(&[1, 9, 2], &[a.clone()], Some(&cfg)).is_err());
        assert!(image_runs(&[1], &[a], None).is_err());
    }
}
