use egui_winit_vulkano::egui::{CollapsingHeader, Context, Visuals};
use egui_winit_vulkano::{egui, Gui};
use nalgebra::{Point3, Vector3};
use std::borrow::BorrowMut;
use std::path::PathBuf;
use std::sync::mpsc::TryRecvError;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;
use vulkano::device::Features;
use vulkano::format::Format;

use structopt::StructOpt;
use vulkano::device::DeviceExtensions;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano_win::VkSurfaceBuild;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event_loop::EventLoop;
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};

use winit::event::{ElementState, Event, KeyboardInput, MouseButton, WindowEvent};
use winit::event_loop::ControlFlow;

use punctum::{
    get_render_pass, load_octree_with_progress_bar, select_physical_device, CameraController,
    CullingMode, Octree, OctreeRenderer, PerspectiveCamera, PointCloud, RenderMode, SHCoefficients,
    SHVertex, SurfaceFrame, Viewport,
};

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,
}

struct GuiState {
    highlight_shs: bool,
    render_mode: RenderMode,
    frustum_culling: Option<CullingMode>,
    lod_threshold: f32,
    debug: bool,
    sh_transparency: bool,

    point_size: u32,
}

impl GuiState {
    fn new(_gui: &mut Gui) -> Self {
        GuiState {
            highlight_shs: false,
            render_mode: RenderMode::Both,
            frustum_culling: Some(CullingMode::Mixed),
            point_size: 1,
            lod_threshold: 0.01,
            debug: false,
            sh_transparency: false,
        }
    }

    pub fn layout(
        &mut self,
        egui_context: Context,
        window: &Window,
        fps: f32,
        camera: &PerspectiveCamera,
        camera_controller: &mut CameraController,
    ) {
        egui_context.set_visuals(Visuals::dark());
        egui::Window::new("Point Cloud Rendering")
            .resizable(false)
            .collapsible(true)
            .show(&egui_context, |ui| {
                CollapsingHeader::new("Rendering")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.checkbox(&mut self.highlight_shs, "Highlight SH Pixels");
                        ui.checkbox(&mut self.debug, "Debug");
                        ui.checkbox(&mut self.sh_transparency, "SH Transparency");

                        ui.horizontal(|ui| {
                            ui.label("Mode:");
                            ui.selectable_value(&mut self.render_mode, RenderMode::Both, "Both");
                            ui.selectable_value(&mut self.render_mode, RenderMode::SHOnly, "SHs");
                            ui.selectable_value(
                                &mut self.render_mode,
                                RenderMode::OctantsOnly,
                                "Octants",
                            );
                        });
                        ui.add(egui::Slider::new(&mut self.point_size, 1..=20).text("Point Size"));
                    });
                CollapsingHeader::new("Culling")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Mode:");
                            ui.selectable_value(&mut self.frustum_culling, None, "Off");
                            ui.selectable_value(
                                &mut self.frustum_culling,
                                Some(CullingMode::Mixed),
                                "Mixed",
                            );
                            ui.selectable_value(
                                &mut self.frustum_culling,
                                Some(CullingMode::SHOnly),
                                "SHs",
                            );
                            ui.selectable_value(
                                &mut self.frustum_culling,
                                Some(CullingMode::OctantsOnly),
                                "Octants",
                            );
                        });
                        ui.add(egui::DragValue::new(&mut self.lod_threshold).speed(0.001));
                    });
            });

        egui::Window::new("Camera")
            .resizable(false)
            .collapsible(true)
            .default_pos([window.inner_size().width as f32, 0.])
            .show(&egui_context, |ui| {
                CollapsingHeader::new("Info")
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::Grid::new("camera grid")
                            .num_columns(2)
                            .spacing([40.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                let pos = camera.position();
                                ui.label("Position");
                                ui.columns(3, |cols| {
                                    for (i, col) in cols.iter_mut().enumerate() {
                                        col.label(format!("{:.3}", pos[i]));
                                    }
                                });
                                ui.end_row();

                                let rot = camera.rotation();
                                ui.label("Rotation");
                                ui.columns(3, |cols| {
                                    for (i, col) in cols.iter_mut().enumerate() {
                                        col.label(format!("{:.3}", rot[i]));
                                    }
                                });
                                ui.end_row();
                                ui.label("FOV (horizontal):");
                                ui.label(format!("{:}", camera.projection.fovy));
                                ui.end_row();

                                ui.label("zNear:");
                                ui.label(format!("{:}", camera.znear()));
                                ui.end_row();
                                ui.label("zFar:");
                                ui.label(format!("{:}", camera.zfar()));
                            });
                    });
                CollapsingHeader::new("Controller")
                    .default_open(true)
                    .show(ui, |ui| {
                        egui::Grid::new("camera controll grid")
                            .num_columns(2)
                            .spacing([40.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Speed:");
                                ui.add(
                                    egui::DragValue::new(&mut camera_controller.speed)
                                        .clamp_range(0. ..=1000.),
                                );
                                ui.end_row();

                                ui.label("Sensivity:");
                                ui.add(
                                    egui::DragValue::new(&mut camera_controller.sensitivity)
                                        .speed(0.01)
                                        .clamp_range(0. ..=100.)
                                        .suffix("°"),
                                );
                                ui.end_row();
                            });
                    });
            });
        let size = window.inner_size();
        egui::Area::new("fps")
            .fixed_pos(egui::pos2(
                size.width as f32 - 0.05 * size.width as f32,
                10.0,
            ))
            .show(&egui_context, |ui| {
                ui.label(format!("{:.2}", fps));
            });
    }
}

fn main() {
    let opt = Opt::from_args();
    let filename = opt.input;

    let mut octree: Octree<f32, f32> = load_octree_with_progress_bar(&filename).unwrap().into();

    // use average color
    for o in octree.borrow_mut().into_iter() {
        let pc: &PointCloud<f32, f32> = o.points().into();
        let (centroid, mean_color) = pc.centroid_and_color();
        o.sh_approximation = Some(SHVertex::new(
            centroid,
            SHCoefficients::new_from_color(mean_color),
        ))
    }

    let octree = Arc::new(octree);

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let mut event_loop = EventLoop::new(); // ignore this for now
    let surface = WindowBuilder::new()
        .with_title("puncTUM")
        .with_inner_size(PhysicalSize::new(800, 600))
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ext_buffer_device_address: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) =
        select_physical_device(&instance, &device_extensions, Some(surface.clone()));

    let (device, mut queues) = Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,

            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            enabled_features: Features {
                shader_int64: true,
                ..Features::none()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    // let swapchain_format = physical_device
    //     .surface_formats(&surface, Default::default())
    //     .unwrap()[0]
    //     .0;

    let swapchain_format = Format::B8G8R8A8_SRGB;

    println!("{:?}", swapchain_format);

    let render_pass = get_render_pass(device.clone(), swapchain_format);

    let mut viewport = Viewport::new(surface.window().inner_size().into());

    let mut frame = SurfaceFrame::new(
        surface.clone(),
        device.clone(),
        physical_device.clone(),
        render_pass.clone(),
        swapchain_format,
    );

    let mut gui = Gui::new(surface.clone(), queue.clone(), true);
    let gui_state = Arc::new(Mutex::new(GuiState::new(&mut gui)));

    let scene_subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let window_size = surface.window().inner_size();
    let aspect_ratio = window_size.width as f32 / window_size.height as f32;
    let mut camera = PerspectiveCamera::new(
        Point3::new(-1.309, -19.021, -28.548),
        Vector3::new(-0.367, 1.367, -0.130),
        aspect_ratio,
    );
    camera.adjust_znear_zfar(octree.bbox());

    let renderer = Arc::new(OctreeRenderer::new(
        device.clone(),
        queue.clone(),
        scene_subpass,
        viewport.clone(),
        octree.clone(),
        &camera,
    ));

    renderer.set_point_size(1);
    renderer.frustum_culling(CullingMode::Mixed, window_size.height as f32);

    let renderer_clone = renderer.clone();
    let gui_state_clone = gui_state.clone();

    let (tx, rx) = mpsc::channel();
    let frustum_culling_thread = thread::spawn(move || loop {
        let (frustum_culling, lod_threshold) = {
            let state = gui_state_clone.lock().unwrap();
            (state.frustum_culling, state.lod_threshold)
        };
        if let Some(culling_mode) = frustum_culling {
            renderer_clone.frustum_culling(culling_mode, lod_threshold);
        }
        match rx.try_recv() {
            Ok(_) | Err(TryRecvError::Disconnected) => {
                break;
            }
            Err(TryRecvError::Empty) => {}
        }
    });

    let mut camera_controller = CameraController::new(50., 1.);

    let mut last_update_inst = Instant::now();

    let mut fps = 0.;
    let mut last_mouse_position: Option<PhysicalPosition<f64>> = None;
    let mut mouse_pressed = false;
    event_loop.run_return(move |event, _, control_flow| match event {
        Event::WindowEvent { event, window_id } if window_id == surface.window().id() => {
            // Update Egui integration so the UI works!
            let _pass_events_to_game = !gui.update(&event);
            if _pass_events_to_game {
                match event {
                    WindowEvent::Resized(size) => {
                        viewport.resize(size.into());
                        renderer.set_viewport(viewport.clone());
                        camera.set_aspect_ratio(size.width as f32 / size.height as f32);
                        frame.force_recreate();
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::CursorMoved {
                        position: new_pos, ..
                    } => {
                        if mouse_pressed {
                            if let Some(last_pos) = last_mouse_position {
                                camera_controller.process_mouse(
                                    (-(new_pos.x - last_pos.x) as f32).into(),
                                    (-(new_pos.y - last_pos.y) as f32).into(),
                                );
                            }
                            last_mouse_position = Some(new_pos);
                        }
                    }
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Left,
                        ..
                    } => match state {
                        ElementState::Pressed => {
                            mouse_pressed = true;
                        }
                        ElementState::Released => {
                            mouse_pressed = false;
                            last_mouse_position = None;
                        }
                    },
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(key),
                                state,
                                ..
                            },
                        ..
                    } => {
                        camera_controller.process_keyboard(key, state);
                    }
                    _ => (),
                }
            }
        }
        Event::RedrawEventsCleared => {
            frame.recreate_if_necessary();

            let time_since_last_frame = last_update_inst.elapsed();
            fps = 1. / time_since_last_frame.as_secs_f32();
            last_update_inst = Instant::now();

            let moved = camera_controller.update_camera(&mut camera, time_since_last_frame);

            if moved {
                camera.adjust_znear_zfar(octree.bbox());
            }

            surface.window().request_redraw();
        }
        Event::RedrawRequested(..) => {
            let mut state = gui_state.lock().unwrap();
            gui.immediate_ui(|gui| {
                let ctx = gui.context();
                state.layout(ctx, surface.window(), fps, &camera, &mut camera_controller);
            });
            renderer.set_point_size(state.point_size);
            renderer.set_highlight_sh(state.highlight_shs);
            renderer.set_transparency(state.sh_transparency);

            renderer.set_camera(&camera);
            renderer.update_uniforms();
            let pc_cb = renderer.render(state.render_mode, state.debug);
            drop(state);

            frame.render(queue.clone(), pc_cb.clone(), &mut gui);
        }
        Event::LoopDestroyed => {
            let _ = tx.send(());
        }
        _ => (),
    });

    frustum_culling_thread.join().unwrap();
}
