use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use pbr::ProgressBar;
use vulkano::device::DeviceExtensions;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano_win::VkSurfaceBuild;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, MouseButton, WindowEvent};
use winit::event_loop::ControlFlow;

use punctum::{
    export_ply, get_render_pass, select_physical_device, CameraController, Octree,
    PerspectiveCamera, PointCloud, PointCloudGPU, PointCloudRenderer, SurfaceFrame, TeeReader,
    Viewport,
};

fn main() {
    let filename = "dataset/octree_64_512max.bin";

    let octree = {
        let in_file = File::open(filename).unwrap();

        let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

        let mut buf = BufReader::new(in_file);

        pb.message(&format!("decoding {}: ", filename));

        pb.set_units(pbr::Units::Bytes);
        let mut tee = TeeReader::new(&mut buf, &mut pb);

        let octree: Octree<f64, u8> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!");

        octree
    };

    let mut pc_raw = PointCloud::from_vec(&octree.into());

    pc_raw.scale_to_size(1000.);

    export_ply(
        &Path::new("test_octree.ply").to_path_buf(),
        &((&pc_raw).into()),
    );

    println!("test: {:?}", pc_raw.points()[0]);

    // let mut pc_raw = PointCloud::from_ply_file("bunny.ply");
    // pc_raw.scale_to_unit_sphere();

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new(); // ignore this for now
    let surface = WindowBuilder::new()
        .with_title("puncTUM")
        .with_inner_size(PhysicalSize::new(512, 512))
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) =
        select_physical_device(&instance, &device_extensions, Some(surface.clone()));

    println!("using device {}", physical_device.properties().device_name);
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

    let mut viewport = Viewport::new(surface.window().inner_size().into());

    let mut frame = SurfaceFrame::new(
        surface.clone(),
        device.clone(),
        physical_device.clone(),
        render_pass.clone(),
        swapchain_format,
    );

    let scene_subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let pc = PointCloudGPU::from_point_cloud(device.clone(), Arc::new(pc_raw));

    let mut renderer = PointCloudRenderer::new(
        device.clone(),
        queue.clone(),
        scene_subpass,
        viewport.clone(),
    );

    let mut camera = PerspectiveCamera::new(); //::look_at(pc.cpu().bounding_box().clone());

    let mut camera_controller = CameraController::new(10., 1.);

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
            viewport.resize(size.into());
            renderer.set_viewport(viewport.clone());
            camera.set_aspect_ratio(size.width as f32 / size.height as f32);
            frame.force_recreate();
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

            let time_since_last_frame = last_update_inst.elapsed();
            let _fps = 1. / time_since_last_frame.as_secs_f32();
            // if fps < 55. {
            //     println!("FPS: {:}", fps);
            // }
            camera_controller.update_camera(&mut camera, time_since_last_frame);

            last_update_inst = Instant::now();
            surface.window().request_redraw();
        }
        Event::RedrawRequested(..) => {
            renderer.set_camera(&camera);
            let pc_cb = renderer.render_point_cloud(queue.clone(), &pc);
            frame.render(queue.clone(), pc_cb.clone());
        }
        _ => (),
    });
}
