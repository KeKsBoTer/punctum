use std::sync::Arc;
use std::time::{Duration, Instant};

use camera::CameraController;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::device::DeviceExtensions;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::swapchain::Surface;
use vulkano_win::VkSurfaceBuild;
use winit::dpi::PhysicalPosition;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, WindowEvent};
use winit::event_loop::ControlFlow;

use pointcloud::PointCloud;
use scene::Scene;

mod camera;
mod pointcloud;
mod renderer;
mod scene;
mod vertex;

fn select_physical_device<'a>(
    instance: &'a Arc<Instance>,
    surface: Arc<Surface<Window>>,
    device_extensions: &DeviceExtensions,
) -> (PhysicalDevice<'a>, QueueFamily<'a>) {
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .expect("no device available");

    (physical_device, queue_family)
}

fn get_render_pass(
    device: Arc<Device>,
    swapchain_format: vulkano::format::Format,
) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain_format,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: vulkano::format::Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap()
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new(); // ignore this for now
    let surface = WindowBuilder::new()
        .with_title("puncTUM")
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) =
        select_physical_device(&instance, surface.clone(), &device_extensions);

    let (device, mut queues) = Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),

            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],

            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let swapchain_format = physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;

    let render_pass = get_render_pass(device.clone(), swapchain_format);

    let mut frame = renderer::Frame::new(
        surface.clone(),
        device.clone(),
        physical_device.clone(),
        render_pass.clone(),
        swapchain_format,
        render_pass.clone(),
    );

    let scene_subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let mut renderer = renderer::PointCloudRenderer::new(device.clone(), scene_subpass);

    let pc = PointCloud::from_ply_file(device.clone(), "bunny.ply");
    let mut scene = Scene::new(pc);

    let mut camera_controller = CameraController::new(0.1, 0.1);

    let mut last_update_inst = Instant::now();

    let mut last_mouse_position: Option<PhysicalPosition<f64>> = None;
    let mut mouse_pressed = false;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            frame.resize(size);
            surface.window().request_redraw();
        }
        Event::WindowEvent {
            event: WindowEvent::CursorMoved {
                position: new_pos, ..
            },
            ..
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
        Event::WindowEvent {
            event:
                WindowEvent::MouseInput {
                    state,
                    button: MouseButton::Left,
                    ..
                },
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
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(key),
                state,
                ..
            }) => {
                camera_controller.process_keyboard(key, state);
            }
            _ => {}
        },

        Event::RedrawEventsCleared => {
            frame.recreate_if_necessary();

            // limit FPS to 60
            let target_frametime = Duration::from_secs_f64(1.0 / 60.0);
            let time_since_last_frame = last_update_inst.elapsed();
            if time_since_last_frame >= target_frametime {
                surface.window().request_redraw();
                println!("FPS: {:}", 1. / time_since_last_frame.as_secs_f32());

                camera_controller.update_camera(&mut scene.camera, time_since_last_frame);

                last_update_inst = Instant::now();
            } else {
                *control_flow = ControlFlow::WaitUntil(
                    Instant::now() + target_frametime - time_since_last_frame,
                );
            }
        }
        Event::RedrawRequested(..) => {
            renderer.render_to_frame(queue.clone(), &scene, &mut frame);
        }
        _ => (),
    });
}
