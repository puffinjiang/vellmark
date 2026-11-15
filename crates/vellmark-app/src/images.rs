//! Lazy viewport-driven image loading, safety validation, base64 decoding,
//! bounded worker decoding (`image` crate), and CPU-side LRU pixel cache (`PRD I-01`, `I-05`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread;

use image::Limits;
use vellmark_render::ImageLoadStatus;

/// Maximum encoded file / clipboard / data-URI image payload (32 MiB).
const MAX_IMAGE_SOURCE_BYTES: usize = 32 * 1024 * 1024;
/// Maximum HTTP(S) download byte cap enforced via `curl`.
const MAX_REMOTE_IMAGE_BYTES: u64 = 16 * 1024 * 1024;
/// Maximum decoded pixel dimensions (defense against decompression bombs).
const MAX_DECODED_DIMENSION: u32 = 8192;
/// Maximum decoded allocation passed to `image::Limits`.
const MAX_DECODED_ALLOC_BYTES: u64 = 128 * 1024 * 1024;
/// Total CPU-side RGBA8 pixel cache budget before LRU eviction.
const MAX_CACHE_PIXEL_BYTES: usize = 128 * 1024 * 1024;
/// Maximum pending queued decode requests.
const MAX_PENDING_JOBS: usize = 64;

/// Decoded RGBA8 pixel buffer cached for GPU texture upload.
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Arc<Vec<u8>>,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    base_dir: Option<PathBuf>,
    source: String,
}

#[derive(Debug, Clone)]
enum CacheEntry {
    Loading { last_used: u64 },
    Loaded { image: DecodedImage, last_used: u64 },
    Failed { last_used: u64 },
}

struct LoadJob {
    key: CacheKey,
    epoch: u64,
}

struct LoadedMessage {
    key: CacheKey,
    epoch: u64,
    result: Option<(u32, u32, Vec<u8>)>,
}

struct WorkerQueue {
    jobs: VecDeque<LoadJob>,
    shutdown: bool,
}

type RedrawNotifier = Arc<dyn Fn() + Send + Sync>;

/// Main-thread image cache coordinating background decoding workers and LRU pixel eviction.
pub struct ImageCache {
    entries: HashMap<CacheKey, CacheEntry>,
    inflight: HashSet<CacheKey>,
    use_seq: u64,
    upload_gen_seq: u64,
    total_pixel_bytes: usize,
    queue: Arc<(Mutex<WorkerQueue>, Condvar)>,
    result_rx: mpsc::Receiver<LoadedMessage>,
    worker_handles: Vec<thread::JoinHandle<()>>,
}

impl Default for ImageCache {
    fn default() -> Self {
        Self::new(None)
    }
}

impl ImageCache {
    /// Creates an image cache backed by a bounded background worker pool.
    /// `on_ready` is invoked from worker threads whenever a decode/fetch finishes.
    pub fn new(on_ready: Option<RedrawNotifier>) -> Self {
        let queue = Arc::new((
            Mutex::new(WorkerQueue {
                jobs: VecDeque::new(),
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let (result_tx, result_rx) = mpsc::channel();
        let mut worker_handles = Vec::with_capacity(2);
        for worker_idx in 0..2 {
            let q = Arc::clone(&queue);
            let tx = result_tx.clone();
            let notifier = on_ready.clone();
            if let Ok(handle) = thread::Builder::new()
                .name(format!("vellmark-img-{worker_idx}"))
                .spawn(move || worker_loop(q, tx, notifier))
            {
                worker_handles.push(handle);
            }
        }
        Self {
            entries: HashMap::new(),
            inflight: HashSet::new(),
            use_seq: 0,
            upload_gen_seq: 0,
            total_pixel_bytes: 0,
            queue,
            result_rx,
            worker_handles,
        }
    }

    /// Drains finished background decodes into the cache and enforces the LRU pixel memory budget.
    /// Returns `true` if any image changed state (caller should request scene refresh).
    pub fn drain_completed(&mut self) -> bool {
        let mut changed = false;
        while let Ok(msg) = self.result_rx.try_recv() {
            let _ = msg.epoch;
            self.inflight.remove(&msg.key);
            self.use_seq = self.use_seq.wrapping_add(1);
            let last_used = self.use_seq;
            if let Some((width, height, rgba)) = msg.result {
                if let Some(CacheEntry::Loaded { image, .. }) = self.entries.get(&msg.key) {
                    self.total_pixel_bytes =
                        self.total_pixel_bytes.saturating_sub(image.rgba.len());
                }
                let byte_len = rgba.len();
                self.total_pixel_bytes += byte_len;
                self.upload_gen_seq = self.upload_gen_seq.wrapping_add(1);
                let image = DecodedImage {
                    width,
                    height,
                    rgba: Arc::new(rgba),
                    generation: self.upload_gen_seq,
                };
                self.entries
                    .insert(msg.key, CacheEntry::Loaded { image, last_used });
            } else {
                if let Some(CacheEntry::Loaded { image, .. }) = self.entries.get(&msg.key) {
                    self.total_pixel_bytes =
                        self.total_pixel_bytes.saturating_sub(image.rgba.len());
                }
                self.entries
                    .insert(msg.key, CacheEntry::Failed { last_used });
            }
            changed = true;
        }
        if changed {
            self.evict_over_budget();
        }
        changed
    }

    /// Queries the current status of `source` relative to `base_dir` without triggering blocking I/O.
    pub fn status_for(
        &self,
        base_dir: Option<&Path>,
        source: &str,
        allow_network_images: bool,
    ) -> ImageLoadStatus {
        if is_remote_url(source) && !allow_network_images {
            return ImageLoadStatus::NetworkBlocked;
        }
        let key = CacheKey {
            base_dir: (!is_self_contained_source(source)).then(|| base_dir.map(Path::to_path_buf)).flatten(),
            source: source.to_string(),
        };
        match self.entries.get(&key) {
            Some(CacheEntry::Loaded { image, .. }) => ImageLoadStatus::Loaded {
                width: image.width,
                height: image.height,
            },
            Some(CacheEntry::Failed { .. }) => ImageLoadStatus::Failed,
            Some(CacheEntry::Loading { .. }) | None => ImageLoadStatus::Pending,
        }
    }

    /// Returns decoded RGBA8 pixels for rendering a visible `SceneImage`.
    pub fn get_decoded(&mut self, base_dir: Option<&Path>, source: &str) -> Option<DecodedImage> {
        let key = CacheKey {
            base_dir: (!is_self_contained_source(source)).then(|| base_dir.map(Path::to_path_buf)).flatten(),
            source: source.to_string(),
        };
        self.use_seq = self.use_seq.wrapping_add(1);
        let seq = self.use_seq;
        if let Some(CacheEntry::Loaded { image, last_used }) = self.entries.get_mut(&key) {
            *last_used = seq;
            return Some(image.clone());
        }
        None
    }

    /// Enqueues any uncached visible viewport images (`PRD I-05`) for async decode/fetch.
    pub fn request_images(
        &mut self,
        base_dir: Option<&Path>,
        sources: &[String],
        allow_network_images: bool,
    ) {
        let mut to_push = Vec::new();
        for src in sources {
            if src.is_empty() || (is_remote_url(src) && !allow_network_images) {
                continue;
            }
            let key = CacheKey {
                base_dir: (!is_self_contained_source(src))
                    .then(|| base_dir.map(Path::to_path_buf))
                    .flatten(),
                source: src.clone(),
            };
            self.use_seq = self.use_seq.wrapping_add(1);
            let seq = self.use_seq;
            match self.entries.get_mut(&key) {
                Some(CacheEntry::Loaded { last_used, .. })
                | Some(CacheEntry::Failed { last_used })
                | Some(CacheEntry::Loading { last_used }) => {
                    *last_used = seq;
                    continue;
                }
                None => {}
            }
            if !self.inflight.contains(&key) {
                self.entries
                    .insert(key.clone(), CacheEntry::Loading { last_used: seq });
                self.inflight.insert(key.clone());
                to_push.push(LoadJob { key, epoch: seq });
            }
        }

        if to_push.is_empty() {
            return;
        }
        let (lock, cvar) = &*self.queue;
        if let Ok(mut guard) = lock.lock() {
            for job in to_push {
                if guard.jobs.len() >= MAX_PENDING_JOBS {
                    guard.jobs.pop_front();
                }
                guard.jobs.push_back(job);
            }
            cvar.notify_all();
        }
    }

    /// Clears cached failed remote requests when toggling `allow_network_images` so remote images reload cleanly.
    pub fn invalidate_remote_entries(&mut self) {
        self.entries.retain(|k, entry| {
            if is_remote_url(&k.source) {
                if let CacheEntry::Loaded { image, .. } = entry {
                    self.total_pixel_bytes =
                        self.total_pixel_bytes.saturating_sub(image.rgba.len());
                }
                false
            } else {
                true
            }
        });
    }

    fn evict_over_budget(&mut self) {
        while self.total_pixel_bytes > MAX_CACHE_PIXEL_BYTES {
            let Some(victim_key) = self
                .entries
                .iter()
                .filter_map(|(k, e)| match e {
                    CacheEntry::Loaded { last_used, .. } => Some((k.clone(), *last_used)),
                    _ => None,
                })
                .min_by_key(|(_, lu)| *lu)
                .map(|(k, _)| k)
            else {
                break;
            };
            if let Some(CacheEntry::Loaded { image, .. }) = self.entries.remove(&victim_key) {
                self.total_pixel_bytes = self.total_pixel_bytes.saturating_sub(image.rgba.len());
            }
        }
    }
}

impl Drop for ImageCache {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.queue;
        if let Ok(mut guard) = lock.lock() {
            guard.shutdown = true;
            cvar.notify_all();
        }
        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

fn worker_loop(
    queue: Arc<(Mutex<WorkerQueue>, Condvar)>,
    tx: mpsc::Sender<LoadedMessage>,
    notifier: Option<RedrawNotifier>,
) {
    let (lock, cvar) = &*queue;
    loop {
        let job = {
            let Ok(mut guard) = lock.lock() else {
                return;
            };
            while !guard.shutdown && guard.jobs.is_empty() {
                let Ok(next_guard) = cvar.wait(guard) else {
                    return;
                };
                guard = next_guard;
            }
            if guard.shutdown {
                return;
            }
            guard.jobs.pop_back()
        };
        let Some(job) = job else {
            continue;
        };
        let result = load_and_decode_image(job.key.base_dir.as_deref(), &job.key.source);
        if tx
            .send(LoadedMessage {
                key: job.key,
                epoch: job.epoch,
                result,
            })
            .is_err()
        {
            return;
        }
        if let Some(cb) = &notifier {
            cb();
        }
    }
}

fn is_remote_url(source: &str) -> bool {
    let lower = source.trim_start().to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

fn is_self_contained_source(source: &str) -> bool {
    let lower = source.trim_start().to_ascii_lowercase();
    lower.starts_with("data:")
        || lower.starts_with("http://")
        || lower.starts_with("https://")
        || lower.starts_with("file://")
}

fn load_and_decode_image(base_dir: Option<&Path>, source: &str) -> Option<(u32, u32, Vec<u8>)> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return None;
    }

    let raw_bytes = if let Some(b64_payload) = parse_data_image_base64(trimmed) {
        decode_base64_str(b64_payload, MAX_IMAGE_SOURCE_BYTES)?
    } else if is_remote_url(trimmed) {
        fetch_remote_image_bytes(trimmed)?
    } else {
        read_local_image_bytes(base_dir, trimmed)?
    };

    decode_raster_rgba8(&raw_bytes)
}

fn parse_data_image_base64(source: &str) -> Option<&str> {
    let after_scheme = source.strip_prefix("data:").or_else(|| {
        source
            .get(..5)
            .filter(|p| p.eq_ignore_ascii_case("data:"))
            .map(|_| &source[5..])
    })?;
    let comma = after_scheme.find(',')?;
    let meta = after_scheme[..comma].trim().to_ascii_lowercase();
    if !meta.starts_with("image/") || !meta.ends_with(";base64") {
        return None;
    }
    let subtype = &meta["image/".len()..meta.len() - ";base64".len()];
    let subtype_clean = subtype.split(';').next().unwrap_or("").trim();
    if !matches!(
        subtype_clean,
        "png" | "jpeg" | "jpg" | "gif" | "webp" | "bmp" | "ico" | "x-icon" | "tiff"
    ) {
        return None;
    }
    Some(after_scheme[comma + 1..].trim())
}

fn fetch_remote_image_bytes(url: &str) -> Option<Vec<u8>> {
    let out = Command::new("curl")
        .args([
            "-fsSL",
            "--max-time",
            "8",
            "--max-filesize",
            &MAX_REMOTE_IMAGE_BYTES.to_string(),
            "--proto",
            "=http,https",
            "--",
            url,
        ])
        .output()
        .ok()?;
    if !out.status.success()
        || out.stdout.is_empty()
        || out.stdout.len() > MAX_IMAGE_SOURCE_BYTES
    {
        return None;
    }
    Some(out.stdout)
}

fn read_local_image_bytes(base_dir: Option<&Path>, source: &str) -> Option<Vec<u8>> {
    let candidate = resolve_local_image_path(base_dir, source)?;
    let meta = fs::metadata(&candidate).ok()?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_IMAGE_SOURCE_BYTES as u64 {
        return None;
    }
    let bytes = fs::read(&candidate).ok()?;
    if bytes.is_empty() || bytes.len() > MAX_IMAGE_SOURCE_BYTES {
        return None;
    }
    Some(bytes)
}

fn resolve_local_image_path(base_dir: Option<&Path>, raw_source: &str) -> Option<PathBuf> {
    if let Some(file_rest) = raw_source.strip_prefix("file://") {
        #[cfg(not(target_os = "windows"))]
        let path_part = file_rest.strip_prefix("localhost").unwrap_or(file_rest);
        #[cfg(target_os = "windows")]
        let path_part = file_rest.trim_start_matches('/');
        let decoded = percent_decode_path(path_part);
        let p = PathBuf::from(decoded);
        return p.is_absolute().then_some(p);
    }

    let raw_p = Path::new(raw_source);
    if raw_p.is_absolute() {
        return Some(raw_p.to_path_buf());
    }
    let base = base_dir?;
    let direct = base.join(raw_p);
    if direct.exists() {
        return Some(direct);
    }
    let decoded = percent_decode_path(raw_source);
    let alt = base.join(decoded);
    Some(alt)
}

fn percent_decode_path(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let (Some(h), Some(l)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2]))
        {
            out.push((h << 4) | l);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_string())
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn decode_raster_rgba8(bytes: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?;
    let mut limits = Limits::default();
    limits.max_image_width = Some(MAX_DECODED_DIMENSION);
    limits.max_image_height = Some(MAX_DECODED_DIMENSION);
    limits.max_alloc = Some(MAX_DECODED_ALLOC_BYTES);
    reader.limits(limits);

    let dyn_img = reader.decode().ok()?;
    let rgba = dyn_img.to_rgba8();
    let (w, h) = rgba.dimensions();
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h, rgba.into_raw()))
}

/// Decodes an RFC 4648 base64 string (ignoring ASCII whitespace) into raw bytes bounded by `max_out_bytes`.
pub fn decode_base64_str(input: &str, max_out_bytes: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity((input.len() / 4).saturating_mul(3).min(max_out_bytes));
    let mut quartet = [0u8; 4];
    let mut qlen = 0usize;
    let mut pad_count = 0usize;

    for b in input.bytes() {
        if b.is_ascii_whitespace() {
            continue;
        }
        if pad_count > 0 && b != b'=' {
            return None;
        }
        let val = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => {
                pad_count += 1;
                if pad_count > 2 || (qlen + pad_count) > 4 {
                    return None;
                }
                0
            }
            _ => return None,
        };
        quartet[qlen] = val;
        qlen += 1;
        if qlen == 4 {
            if pad_count == 0 {
                if out.len() + 3 > max_out_bytes {
                    return None;
                }
                out.push((quartet[0] << 2) | (quartet[1] >> 4));
                out.push((quartet[1] << 4) | (quartet[2] >> 2));
                out.push((quartet[2] << 6) | quartet[3]);
            } else if pad_count == 1 {
                if out.len() + 2 > max_out_bytes {
                    return None;
                }
                out.push((quartet[0] << 2) | (quartet[1] >> 4));
                out.push((quartet[1] << 4) | (quartet[2] >> 2));
            } else if pad_count == 2 {
                if out.len() + 1 > max_out_bytes {
                    return None;
                }
                out.push((quartet[0] << 2) | (quartet[1] >> 4));
            }
            qlen = 0;
            pad_count = 0;
        }
    }
    if qlen != 0 {
        return None;
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn make_test_png(w: u32, h: u32) -> Vec<u8> {
        let mut img = image::RgbaImage::new(w, h);
        for (x, y, px) in img.enumerate_pixels_mut() {
            *px = image::Rgba([(x * 13) as u8, (y * 17) as u8, 220, 255]);
        }
        let mut buf = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    fn wait_for_cache(cache: &mut ImageCache, base: Option<&Path>, src: &str, allow_net: bool) -> ImageLoadStatus {
        for _ in 0..100 {
            cache.drain_completed();
            let st = cache.status_for(base, src, allow_net);
            if st != ImageLoadStatus::Pending {
                return st;
            }
            thread::sleep(Duration::from_millis(10));
        }
        cache.status_for(base, src, allow_net)
    }

    #[test]
    fn test_image_cache_local_file_and_data_uri_and_network_policy() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("vellmark-img-cache-{nanos}"));
        let assets_dir = dir.join("assets");
        fs::create_dir_all(&assets_dir).unwrap();
        let png_bytes = make_test_png(16, 8);
        let disk_path = assets_dir.join("my chart.png");
        fs::write(&disk_path, &png_bytes).unwrap();

        let woke = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let woke_clone = Arc::clone(&woke);
        let mut cache = ImageCache::new(Some(Arc::new(move || {
            woke_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })));

        // 1. Relative percent-encoded local path decodes asynchronously & fires redraw callback.
        let rel = "./assets/my%20chart.png".to_string();
        assert_eq!(cache.status_for(Some(&dir), &rel, true), ImageLoadStatus::Pending);
        cache.request_images(Some(&dir), std::slice::from_ref(&rel), true);
        assert_eq!(
            wait_for_cache(&mut cache, Some(&dir), &rel, true),
            ImageLoadStatus::Loaded {
                width: 16,
                height: 8,
            }
        );
        let dec = cache.get_decoded(Some(&dir), &rel).expect("decoded pixels");
        assert_eq!(dec.rgba.len(), 16 * 8 * 4);
        assert!(woke.load(std::sync::atomic::Ordering::Relaxed) >= 1);

        // 2. Inline base64 data URI decodes without disk access.
        let encoded = base64_encode_for_test(&png_bytes);
        let data_uri = format!("data:image/png;base64,{encoded}");
        cache.request_images(None, std::slice::from_ref(&data_uri), false);
        assert_eq!(
            wait_for_cache(&mut cache, None, &data_uri, false),
            ImageLoadStatus::Loaded {
                width: 16,
                height: 8,
            }
        );

        // 3. Remote URL honors allow_network_images = false cleanly.
        assert_eq!(
            cache.status_for(Some(&dir), "https://example.com/a.png", false),
            ImageLoadStatus::NetworkBlocked
        );

        // 4. Non-existent local file resolves to Failed without panicking.
        let missing = "./assets/does-not-exist.png".to_string();
        cache.request_images(Some(&dir), std::slice::from_ref(&missing), true);
        assert_eq!(
            wait_for_cache(&mut cache, Some(&dir), &missing, true),
            ImageLoadStatus::Failed
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_decode_base64_str_valid_and_invalid() {
        assert_eq!(decode_base64_str("SGVsbG8=", 64), Some(b"Hello".to_vec()));
        assert_eq!(decode_base64_str("SGVs\n bG8h", 64), Some(b"Hello!".to_vec()));
        assert_eq!(decode_base64_str("abc", 64), None);
        assert_eq!(decode_base64_str("SGVsbG8=", 2), None);
        assert_eq!(decode_base64_str("SGVs=G8=", 64), None);
    }

    fn base64_encode_for_test(input: &[u8]) -> String {
        const TBL: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in input.chunks(3) {
            let b0 = chunk[0] as usize;
            let b1 = *chunk.get(1).unwrap_or(&0) as usize;
            let b2 = *chunk.get(2).unwrap_or(&0) as usize;
            out.push(TBL[b0 >> 2] as char);
            out.push(TBL[((b0 & 0x03) << 4) | (b1 >> 4)] as char);
            out.push(if chunk.len() > 1 {
                TBL[((b1 & 0x0f) << 2) | (b2 >> 6)] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                TBL[b2 & 0x3f] as char
            } else {
                '='
            });
        }
        out
    }
}
