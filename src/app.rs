use std::borrow::Cow;
use std::collections::HashSet;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::math::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    camera_pos: [f32; 4],
    camera_basis: [[f32; 4]; 4],
    view_normal: [f32; 4],
    proj_params: [f32; 4],
    time_and_flags: [f32; 4],
    resolution: [f32; 4],
}

pub struct App {
    window: Option<Window>,
    size: winit::dpi::PhysicalSize<u32>,
    instance: wgpu::Instance,
    surface: Option<wgpu::Surface<'static>>,
    adapter: Option<wgpu::Adapter>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    config: Option<wgpu::SurfaceConfiguration>,
    pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,

    // Camera position is stored as [x, y, z, w] and interpreted as the
    // position of the camera in 4-space. NOTE: the shader's center-of-screen
    // gaze is the normalized diagonal +(1,1,1,1)/2, so to have the scene
    // centered at startup we spawn 8 units out along the REVERSE diagonal:
    // -(1,1,1,1)/2 * 8 = [-4, -4, -4, -4]. This is outside the sphere
    // (radius 1.5 at the origin) and outside the tesseract
    // (centered at [3, 0, 0, 0]).
    world_offset: Vec4,
    cam_basis: Mat4,
    keys_pressed: HashSet<KeyCode>,
    show_sphere: bool,
    show_tesseract: bool,
    time: f32,

    // Timestamp of the previous frame, for real delta-time pacing.
    last_frame: Option<Instant>,
}

impl App {
    pub fn new() -> Self {
        App {
            window: None,
            size: winit::dpi::PhysicalSize::new(800, 600),
            instance: wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                dx12_shader_compiler: wgpu::Dx12Compiler::Fxc,
                gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
                flags: wgpu::InstanceFlags::empty(),
            }),
            surface: None,
            adapter: None,
            device: None,
            queue: None,
            config: None,
            pipeline: None,
            uniform_buffer: None,
            bind_group: None,
            world_offset: [-4.0, -4.0, -4.0, -4.0],
            cam_basis: mat4_identity(),
            keys_pressed: HashSet::new(),
            show_sphere: true,
            show_tesseract: true,
            time: 0.0,
            last_frame: None,
        }
    }

    async fn init_gpu(&mut self, window: &Window) {
        let size = window.inner_size();
        self.size = size;

        let surface = self.instance.create_surface(window).unwrap();
        let surface_static = unsafe {
            std::mem::transmute::<wgpu::Surface<'_>, wgpu::Surface<'static>>(surface)
        };
        self.surface = Some(surface_static);

        let adapter = self.instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: self.surface.as_ref(),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");
        self.adapter = Some(adapter);

        let adapter = self.adapter.as_ref().unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        self.device = Some(device);
        self.queue = Some(queue);

        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();

        let surface_caps = surface.get_capabilities(adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(device, &config);
        self.config = Some(config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shader.wgsl"))),
        });

        let initial_uniforms = Uniforms {
            camera_pos: self.world_offset,
            camera_basis: self.cam_basis,
            view_normal: [1.0, 1.0, 1.0, 1.0],
            proj_params: [1.0, 0.0, 0.0, 0.0],
            time_and_flags: [0.0, 1.0, 1.0, 0.0],
            resolution: [size.width as f32, size.height as f32, 0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[initial_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        self.uniform_buffer = Some(uniform_buffer);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: self.uniform_buffer.as_ref().unwrap(),
                    offset: 0,
                    size: None,
                }),
            }],
            label: Some("bind_group"),
        });
        self.bind_group = Some(bind_group);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_compilation_options = wgpu::PipelineCompilationOptions::default();
        let fragment_compilation_options = wgpu::PipelineCompilationOptions::default();

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: vertex_compilation_options,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.config.as_ref().unwrap().format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: fragment_compilation_options,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });
        self.pipeline = Some(pipeline);
    }

    fn update_camera(&mut self, dt: f32) {
        // Speeds are per SECOND now (dt is real elapsed time, clamped by the
        // caller). At 60 FPS these are equivalent to the old per-frame values
        // of 0.05 / 0.02, but they no longer depend on how fast the poll loop runs.
        let speed = 3.0 * dt;       // world units per second
        let rot_speed = 1.2 * dt;   // radians per second

        // Translations (slide along all four axes):
        //   W/S = -z/+z, A/D = -y/+y, Q/E = -x/+x, Z/X = -w/+w
        if self.keys_pressed.contains(&KeyCode::KeyW) { self.world_offset[2] -= speed; }
        if self.keys_pressed.contains(&KeyCode::KeyS) { self.world_offset[2] += speed; }
        if self.keys_pressed.contains(&KeyCode::KeyA) { self.world_offset[1] -= speed; }
        if self.keys_pressed.contains(&KeyCode::KeyD) { self.world_offset[1] += speed; }
        if self.keys_pressed.contains(&KeyCode::KeyQ) { self.world_offset[0] -= speed; }
        if self.keys_pressed.contains(&KeyCode::KeyE) { self.world_offset[0] += speed; }
        if self.keys_pressed.contains(&KeyCode::KeyZ) { self.world_offset[3] -= speed; }
        if self.keys_pressed.contains(&KeyCode::KeyX) { self.world_offset[3] += speed; }

        // Rotations: all six 4D rotation planes, one pair of keys each.
        // Pre-multiplying by rot4() applies the rotation to the camera's
        // basis rows (local/camera axes).
        if self.keys_pressed.contains(&KeyCode::ArrowUp) {
            self.cam_basis = mat4_mul(&rot4(0, 3, rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::ArrowDown) {
            self.cam_basis = mat4_mul(&rot4(0, 3, -rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::ArrowLeft) {
            self.cam_basis = mat4_mul(&rot4(1, 3, rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::ArrowRight) {
            self.cam_basis = mat4_mul(&rot4(1, 3, -rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::KeyR) {
            self.cam_basis = mat4_mul(&rot4(0, 1, rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::KeyT) {
            self.cam_basis = mat4_mul(&rot4(0, 1, -rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::KeyF) {
            self.cam_basis = mat4_mul(&rot4(0, 2, rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::KeyG) {
            self.cam_basis = mat4_mul(&rot4(0, 2, -rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::KeyV) {
            self.cam_basis = mat4_mul(&rot4(1, 2, rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::KeyB) {
            self.cam_basis = mat4_mul(&rot4(1, 2, -rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::KeyN) {
            self.cam_basis = mat4_mul(&rot4(2, 3, rot_speed), &self.cam_basis);
        }
        if self.keys_pressed.contains(&KeyCode::KeyM) {
            self.cam_basis = mat4_mul(&rot4(2, 3, -rot_speed), &self.cam_basis);
        }
    }

    fn update_uniforms(&mut self) {
        if self.uniform_buffer.is_none() || self.queue.is_none() {
            return;
        }

        let uniforms = Uniforms {
            camera_pos: self.world_offset,
            camera_basis: self.cam_basis,
            view_normal: [1.0, 1.0, 1.0, 1.0],
            proj_params: [1.0, 0.0, 0.0, 0.0],
            time_and_flags: [
                self.time,
                if self.show_sphere { 1.0 } else { 0.0 },
                if self.show_tesseract { 1.0 } else { 0.0 },
                0.0,
            ],
            resolution: [self.size.width as f32, self.size.height as f32, 0.0, 0.0],
        };

        self.queue.as_ref().unwrap().write_buffer(
            self.uniform_buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[uniforms]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if self.pipeline.is_none() || self.bind_group.is_none() || self.surface.is_none() || self.config.is_none() {
            return Ok(());
        }

        let output = self.surface.as_ref().unwrap().get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.as_ref().unwrap().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.04,
                            g: 0.04,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(self.pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.as_ref().unwrap().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop
                .create_window(
                    winit::window::Window::default_attributes()
                        .with_title("4D Raymarcher (Zoomed 4-Point Perspective) — 1/2: Toggle | WASD/QE/ZX: Move | Arrows/RT/FG/VB/NM: Rotate")
                        .with_inner_size(winit::dpi::LogicalSize::new(800, 600)),
                )
                .unwrap();

            pollster::block_on(self.init_gpu(&window));
            self.window = Some(window);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.size = size;
                if size.width > 0 && size.height > 0 {
                    if let (Some(device), Some(surface), Some(config)) = (&self.device, &self.surface, &self.config) {
                        let mut new_config = config.clone();
                        new_config.width = size.width;
                        new_config.height = size.height;
                        surface.configure(device, &new_config);
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.keys_pressed.insert(key);
                            match key {
                                KeyCode::Digit1 => self.show_sphere = !self.show_sphere,
                                KeyCode::Digit2 => self.show_tesseract = !self.show_tesseract,
                                KeyCode::Escape => event_loop.exit(),
                                _ => {}
                            }
                        }
                        ElementState::Released => {
                            self.keys_pressed.remove(&key);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // Real delta-time pacing: dt is measured with Instant and clamped to
        // 0.1 s so that a stall (debugger pause, window drag, slow frame)
        // doesn't teleport the camera when it resumes.
        let now = Instant::now();
        let dt = self
            .last_frame
            .map(|last| (now - last).as_secs_f32())
            .unwrap_or(1.0 / 60.0)
            .min(0.1);
        self.last_frame = Some(now);

        self.time += dt;
        self.update_camera(dt);
        self.update_uniforms();
        if let Err(e) = self.render() {
            eprintln!("{:?}", e);
        }
    }
}
