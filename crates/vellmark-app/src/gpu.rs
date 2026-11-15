//! GPU hardware-accelerated 2D renderer using `wgpu 30` (Metal/Vulkan/DX12),
//! `cosmic_text::SwashCache` glyph atlas rasterization, and SDF rounded-rect/glyph instancing.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use cosmic_text::{CacheKey, FontSystem, SwashCache, SwashContent};
use vellmark_render::{DrawList, RgbaColor, SceneGlyph, SceneImage, SceneRect};
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::images::DecodedImage;

const ATLAS_SIZE: u32 = 2048;
const MAX_GPU_IMAGE_BYTES: usize = 128 * 1024 * 1024;
const MAX_GPU_IMAGE_COUNT: usize = 64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GlobalsUniform {
    viewport_size: [f32; 2],
    is_srgb_target: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuadInstance {
    pos_size: [f32; 4],
    uv_rect: [f32; 4],
    color: [f32; 4],
    border_color: [f32; 4],
    clip_rect: [f32; 4],
    params: [f32; 4], // [corner_radius, border_width, mode (0=rect, 1=alpha_glyph, 2=color_glyph), 0.0]
}

impl QuadInstance {
    fn attrs() -> &'static [wgpu::VertexAttribute] {
        &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 32,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 48,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 64,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x4,
            },
            wgpu::VertexAttribute {
                offset: 80,
                shader_location: 5,
                format: wgpu::VertexFormat::Float32x4,
            },
        ]
    }
}

#[derive(Debug, Clone, Copy)]
struct AtlasEntry {
    uv: [f32; 4],
    offset_x: i32,
    offset_y: i32,
    width: u32,
    height: u32,
    is_color: bool,
}

struct GlyphAtlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    swash: SwashCache,
    entries: HashMap<CacheKey, Option<AtlasEntry>>,
    cursor_x: u32,
    cursor_y: u32,
    row_height: u32,
}

impl GlyphAtlas {
    fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vellmark_glyph_atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            swash: SwashCache::new(),
            entries: HashMap::with_capacity(1024),
            cursor_x: 1,
            cursor_y: 1,
            row_height: 0,
        }
    }

    fn get_or_upload(
        &mut self,
        queue: &wgpu::Queue,
        font_system: &mut FontSystem,
        key: CacheKey,
    ) -> Option<AtlasEntry> {
        if let Some(existing) = self.entries.get(&key) {
            return *existing;
        }
        let img = match self.swash.get_image_uncached(font_system, key) {
            Some(img) if img.placement.width > 0 && img.placement.height > 0 => img,
            _ => {
                self.entries.insert(key, None);
                return None;
            }
        };

        let w = img.placement.width;
        let h = img.placement.height;
        if w + 2 > ATLAS_SIZE || h + 2 > ATLAS_SIZE {
            self.entries.insert(key, None);
            return None;
        }

        if self.cursor_x + w + 1 > ATLAS_SIZE {
            self.cursor_x = 1;
            self.cursor_y += self.row_height + 1;
            self.row_height = 0;
        }
        if self.cursor_y + h + 1 > ATLAS_SIZE {
            // Reset shelf packer cleanly if texture fills across many zooms/fonts.
            self.entries.clear();
            self.cursor_x = 1;
            self.cursor_y = 1;
            self.row_height = 0;
        }

        let (ax, ay) = (self.cursor_x, self.cursor_y);
        self.cursor_x += w + 1;
        self.row_height = self.row_height.max(h);

        let (rgba, is_color) = match img.content {
            SwashContent::Mask => {
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for &alpha in &img.data {
                    out.extend_from_slice(&[255, 255, 255, alpha]);
                }
                (out, false)
            }
            SwashContent::SubpixelMask => {
                let mut out = Vec::with_capacity((w * h * 4) as usize);
                for chunk in img.data.as_chunks::<3>().0 {
                    let lum =
                        (0.299 * chunk[0] as f32 + 0.587 * chunk[1] as f32 + 0.114 * chunk[2] as f32)
                            .round()
                            .clamp(0.0, 255.0) as u8;
                    out.extend_from_slice(&[255, 255, 255, lum]);
                }
                (out, false)
            }
            SwashContent::Color => (img.data, true),
        };

        if rgba.len() == (w * h * 4) as usize {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: ax, y: ay, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
        }

        let inv = 1.0 / ATLAS_SIZE as f32;
        let entry = AtlasEntry {
            uv: [
                ax as f32 * inv,
                ay as f32 * inv,
                (ax + w) as f32 * inv,
                (ay + h) as f32 * inv,
            ],
            offset_x: img.placement.left,
            offset_y: img.placement.top,
            width: w,
            height: h,
            is_color,
        };
        self.entries.insert(key, Some(entry));
        Some(entry)
    }
}

struct GpuImageEntry {
    generation: u64,
    byte_size: usize,
    last_used_frame: u64,
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone)]
enum DrawBatch {
    Atlas(Range<u32>),
    Image(String, Range<u32>),
}

/// Hardware-accelerated `wgpu` renderer owning the window swapchain surface and instanced SDF pipeline.
pub struct WgpuRenderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    image_sampler: wgpu::Sampler,
    atlas: GlyphAtlas,
    image_textures: HashMap<String, GpuImageEntry>,
    image_texture_bytes: usize,
    frame_seq: u64,
    instance_buf: wgpu::Buffer,
    instance_capacity: usize,
    staging_instances: Vec<QuadInstance>,
}

impl WgpuRenderer {
    /// Initialize GPU instance, surface, adapter, device, and pipeline for `window`.
    pub fn new(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .map_err(|e| format!("failed to create wgpu surface: {e}"))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .map_err(|e| format!("failed to request GPU adapter: {e}"))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vellmark_device"),
            ..Default::default()
        }))
        .map_err(|e| format!("failed to request GPU device: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        let preferred_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| "surface reported no supported formats".to_string())?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .ok_or_else(|| "surface incompatible with adapter".to_string())?;
        config.format = preferred_format;
        surface.configure(&device, &config);

        let atlas = GlyphAtlas::new(&device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vellmark_glyph_sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vellmark_image_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let init_uniform = GlobalsUniform {
            viewport_size: [width as f32, height as f32],
            is_srgb_target: if config.format.is_srgb() { 1.0 } else { 0.0 },
            _pad: 0.0,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vellmark_globals"),
            contents: bytemuck::bytes_of(&init_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vellmark_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vellmark_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vellmark_ui_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(QUAD_SHADER_WGSL)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vellmark_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vellmark_quad_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<QuadInstance>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: QuadInstance::attrs(),
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let initial_cap = 4096usize;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vellmark_instances"),
            size: (initial_cap * std::mem::size_of::<QuadInstance>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buf,
            bind_group_layout,
            bind_group,
            image_sampler,
            atlas,
            image_textures: HashMap::new(),
            image_texture_bytes: 0,
            frame_seq: 0,
            instance_buf,
            instance_capacity: initial_cap,
            staging_instances: Vec::with_capacity(initial_cap),
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Upload and render layered rects, decoded images, and glyphs in physical pixels via GPU instancing.
    pub fn render_layers<F>(
        &mut self,
        clear_color: RgbaColor,
        scale_factor: f32,
        font_system: &mut FontSystem,
        layers: &[&DrawList],
        mut resolve_image: F,
    ) -> Result<(), String>
    where
        F: FnMut(&str) -> Option<DecodedImage>,
    {
        let scale = scale_factor.max(0.5);
        self.frame_seq = self.frame_seq.wrapping_add(1);
        self.staging_instances.clear();
        let mut batches: Vec<DrawBatch> = Vec::new();

        for layer in layers {
            let rects_start = self.staging_instances.len() as u32;
            for rect in &layer.rects {
                self.push_rect_instance(rect, scale);
            }
            let rects_end = self.staging_instances.len() as u32;
            if rects_end > rects_start {
                push_atlas_batch(&mut batches, rects_start..rects_end);
            }

            for img in &layer.images {
                if self.push_image_instance(img, scale, &mut resolve_image) {
                    let end = self.staging_instances.len() as u32;
                    batches.push(DrawBatch::Image(img.source.clone(), (end - 1)..end));
                }
            }

            let glyphs_start = self.staging_instances.len() as u32;
            for glyph in &layer.glyphs {
                self.push_glyph_instance(glyph, scale, font_system);
            }
            let glyphs_end = self.staging_instances.len() as u32;
            if glyphs_end > glyphs_start {
                push_atlas_batch(&mut batches, glyphs_start..glyphs_end);
            }
        }
        self.evict_image_textures_if_needed();

        let uniform = GlobalsUniform {
            viewport_size: [self.config.width as f32, self.config.height as f32],
            is_srgb_target: if self.config.format.is_srgb() { 1.0 } else { 0.0 },
            _pad: 0.0,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));

        let instance_count = self.staging_instances.len();
        if instance_count > self.instance_capacity {
            let new_cap = instance_count.next_power_of_two().max(4096);
            self.instance_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vellmark_instances"),
                size: (new_cap * std::mem::size_of::<QuadInstance>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_capacity = new_cap;
        }
        if instance_count > 0 {
            self.queue.write_buffer(
                &self.instance_buf,
                0,
                bytemuck::cast_slice(&self.staging_instances),
            );
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex) => tex,
            wgpu::CurrentSurfaceTexture::Suboptimal(tex) => {
                self.surface.configure(&self.device, &self.config);
                tex
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                return Err("surface validation error acquiring swapchain texture".to_string());
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vellmark_frame_encoder"),
            });

        let clear = if self.config.format.is_srgb() {
            let [r, g, b, a] = clear_color.to_f32_array();
            wgpu::Color {
                r: srgb_channel_to_linear(r as f64),
                g: srgb_channel_to_linear(g as f64),
                b: srgb_channel_to_linear(b as f64),
                a: a as f64,
            }
        } else {
            let [r, g, b, a] = clear_color.to_f32_array();
            wgpu::Color {
                r: r as f64,
                g: g as f64,
                b: b as f64,
                a: a as f64,
            }
        };

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("vellmark_main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            if instance_count > 0 {
                pass.set_vertex_buffer(0, self.instance_buf.slice(..));
                for batch in &batches {
                    match batch {
                        DrawBatch::Atlas(range) => {
                            pass.set_bind_group(0, &self.bind_group, &[]);
                            pass.draw(0..6, range.start..range.end);
                        }
                        DrawBatch::Image(source, range) => {
                            if let Some(entry) = self.image_textures.get(source) {
                                pass.set_bind_group(0, &entry.bind_group, &[]);
                                pass.draw(0..6, range.start..range.end);
                            }
                        }
                    }
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(output);
        Ok(())
    }

    fn push_rect_instance(&mut self, r: &SceneRect, scale: f32) {
        if r.rect.w <= 0.0 || r.rect.h <= 0.0 {
            return;
        }
        let clip = r
            .clip
            .map(|c| [c.x * scale, c.y * scale, c.w * scale, c.h * scale])
            .unwrap_or([0.0, 0.0, -1.0, -1.0]);
        self.staging_instances.push(QuadInstance {
            pos_size: [
                r.rect.x * scale,
                r.rect.y * scale,
                r.rect.w * scale,
                r.rect.h * scale,
            ],
            uv_rect: [0.0; 4],
            color: r.color.to_f32_array(),
            border_color: r.border_color.to_f32_array(),
            clip_rect: clip,
            params: [r.corner_radius * scale, r.border_width * scale, 0.0, 0.0],
        });
    }

    fn push_glyph_instance(
        &mut self,
        g: &SceneGlyph,
        scale: f32,
        font_system: &mut FontSystem,
    ) {
        let Some(entry) = self.atlas.get_or_upload(font_system_queue(&self.queue), font_system, g.cache_key) else {
            return;
        };
        let clip = g
            .clip
            .map(|c| [c.x * scale, c.y * scale, c.w * scale, c.h * scale])
            .unwrap_or([0.0, 0.0, -1.0, -1.0]);
        let gx = (g.x + entry.offset_x) as f32;
        let gy = (g.y - entry.offset_y) as f32;
        self.staging_instances.push(QuadInstance {
            pos_size: [gx, gy, entry.width as f32, entry.height as f32],
            uv_rect: entry.uv,
            color: g.color.to_f32_array(),
            border_color: [0.0; 4],
            clip_rect: clip,
            params: [0.0, 0.0, if entry.is_color { 2.0 } else { 1.0 }, 0.0],
        });
    }

    fn push_image_instance<F>(
        &mut self,
        img: &SceneImage,
        scale: f32,
        resolve_image: &mut F,
    ) -> bool
    where
        F: FnMut(&str) -> Option<DecodedImage>,
    {
        if img.rect.w <= 0.0 || img.rect.h <= 0.0 {
            return false;
        }
        let Some(decoded) = resolve_image(&img.source) else {
            return false;
        };
        if !self.ensure_image_texture(&img.source, &decoded) {
            return false;
        }
        let clip = img
            .clip
            .map(|c| [c.x * scale, c.y * scale, c.w * scale, c.h * scale])
            .unwrap_or([0.0, 0.0, -1.0, -1.0]);
        self.staging_instances.push(QuadInstance {
            pos_size: [
                img.rect.x * scale,
                img.rect.y * scale,
                img.rect.w * scale,
                img.rect.h * scale,
            ],
            uv_rect: [0.0, 0.0, 1.0, 1.0],
            color: [1.0, 1.0, 1.0, img.opacity.clamp(0.0, 1.0)],
            border_color: [0.0; 4],
            clip_rect: clip,
            params: [img.corner_radius * scale, 0.0, 3.0, 0.0],
        });
        true
    }

    fn ensure_image_texture(&mut self, source: &str, decoded: &DecodedImage) -> bool {
        let expected_bytes = (decoded.width as usize)
            .saturating_mul(decoded.height as usize)
            .saturating_mul(4);
        if decoded.width == 0 || decoded.height == 0 || decoded.rgba.len() != expected_bytes {
            return false;
        }
        let frame_seq = self.frame_seq;
        if let Some(existing) = self.image_textures.get_mut(source)
            && existing.generation == decoded.generation
        {
            existing.last_used_frame = frame_seq;
            return true;
        }

        if let Some(old) = self.image_textures.remove(source) {
            self.image_texture_bytes = self.image_texture_bytes.saturating_sub(old.byte_size);
        }

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vellmark_image_tex"),
            size: wgpu::Extent3d {
                width: decoded.width,
                height: decoded.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &decoded.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(decoded.width * 4),
                rows_per_image: Some(decoded.height),
            },
            wgpu::Extent3d {
                width: decoded.width,
                height: decoded.height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vellmark_image_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.image_sampler),
                },
            ],
        });

        self.image_texture_bytes += expected_bytes;
        self.image_textures.insert(
            source.to_string(),
            GpuImageEntry {
                generation: decoded.generation,
                byte_size: expected_bytes,
                last_used_frame: frame_seq,
                _texture: texture,
                bind_group,
            },
        );
        true
    }

    fn evict_image_textures_if_needed(&mut self) {
        while self.image_textures.len() > MAX_GPU_IMAGE_COUNT
            || self.image_texture_bytes > MAX_GPU_IMAGE_BYTES
        {
            let Some(victim) = self
                .image_textures
                .iter()
                .filter(|(_, e)| e.last_used_frame < self.frame_seq)
                .min_by_key(|(_, e)| e.last_used_frame)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            if let Some(old) = self.image_textures.remove(&victim) {
                self.image_texture_bytes =
                    self.image_texture_bytes.saturating_sub(old.byte_size);
            }
        }
    }
}

fn push_atlas_batch(batches: &mut Vec<DrawBatch>, range: Range<u32>) {
    if let Some(DrawBatch::Atlas(last)) = batches.last_mut()
        && last.end == range.start
    {
        last.end = range.end;
        return;
    }
    batches.push(DrawBatch::Atlas(range));
}

#[inline]
fn font_system_queue(q: &wgpu::Queue) -> &wgpu::Queue {
    q
}

fn srgb_channel_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

const QUAD_SHADER_WGSL: &str = r#"
struct Globals {
    viewport_size: vec2<f32>,
    is_srgb_target: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;

struct InstanceInput {
    @location(0) pos_size: vec4<f32>,
    @location(1) uv_rect: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) border_color: vec4<f32>,
    @location(4) clip_rect: vec4<f32>,
    @location(5) params: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_px: vec2<f32>,
    @location(1) rect_size: vec2<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) border_color: vec4<f32>,
    @location(5) clip_rect: vec4<f32>,
    @location(6) frag_px: vec2<f32>,
    @location(7) params: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, inst: InstanceInput) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    let unit = corners[vid];
    let local_px = unit * inst.pos_size.zw;
    let frag_px = inst.pos_size.xy + local_px;
    let ndc = (frag_px / max(globals.viewport_size, vec2<f32>(1.0, 1.0))) * 2.0 - 1.0;

    var out: VertexOutput;
    out.clip_pos = vec4<f32>(ndc.x, -ndc.y, 0.0, 1.0);
    out.local_px = local_px;
    out.rect_size = inst.pos_size.zw;
    out.uv = mix(inst.uv_rect.xy, inst.uv_rect.zw, unit);
    out.color = inst.color;
    out.border_color = inst.border_color;
    out.clip_rect = inst.clip_rect;
    out.frag_px = frag_px;
    out.params = inst.params;
    return out;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, c <= vec3<f32>(0.04045));
}

fn rounded_edge_dist(local_px: vec2<f32>, rect_size: vec2<f32>, raw_radius: f32) -> f32 {
    let half_size = rect_size * 0.5;
    let radius = min(raw_radius, min(half_size.x, half_size.y));
    let q = abs(local_px - half_size) - (half_size - vec2<f32>(radius));
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(atlas_tex, atlas_sampler, in.uv);

    if (in.clip_rect.z > 0.0) {
        if (in.frag_px.x < in.clip_rect.x
            || in.frag_px.x > in.clip_rect.x + in.clip_rect.z
            || in.frag_px.y < in.clip_rect.y
            || in.frag_px.y > in.clip_rect.y + in.clip_rect.w)
        {
            discard;
        }
    }

    let mode = in.params.z;
    var out_col: vec4<f32>;

    if (mode > 2.5) {
        let dist = rounded_edge_dist(in.local_px, in.rect_size, in.params.x);
        let edge_alpha = 1.0 - smoothstep(-0.5, 0.5, dist);
        let rgb = select(texel.rgb, srgb_to_linear(texel.rgb), globals.is_srgb_target > 0.5);
        out_col = vec4<f32>(rgb, texel.a * in.color.a * edge_alpha);
    } else if (mode > 1.5) {
        out_col = vec4<f32>(texel.rgb, texel.a * in.color.a);
    } else if (mode > 0.5) {
        let rgb = select(in.color.rgb, srgb_to_linear(in.color.rgb), globals.is_srgb_target > 0.5);
        out_col = vec4<f32>(rgb, in.color.a * texel.a);
    } else {
        let dist = rounded_edge_dist(in.local_px, in.rect_size, in.params.x);
        let edge_alpha = 1.0 - smoothstep(-0.5, 0.5, dist);
        var fill_col = in.color;
        let bw = in.params.y;
        if (bw > 0.0) {
            let border_mix = smoothstep(-bw - 0.5, -bw + 0.5, dist);
            fill_col = mix(in.color, in.border_color, border_mix);
        }
        let rgb = select(fill_col.rgb, srgb_to_linear(fill_col.rgb), globals.is_srgb_target > 0.5);
        out_col = vec4<f32>(rgb, fill_col.a * edge_alpha);
    }

    if (out_col.a <= 0.001) {
        discard;
    }
    return out_col;
}
"#;
