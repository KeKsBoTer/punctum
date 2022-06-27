use nalgebra::{Point3, Vector3, Vector4};
use std::borrow::BorrowMut;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::sync::mpsc::TryRecvError;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;
use vulkano::device::Features;

use pbr::ProgressBar;
use structopt::StructOpt;
use vulkano::device::DeviceExtensions;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano_win::VkSurfaceBuild;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event_loop::EventLoop;
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::WindowBuilder;

use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;

use punctum::{
    get_render_pass, select_physical_device, CameraController, Octree, OctreeRenderer,
    PerspectiveCamera, PointCloud, SHVertex, SurfaceFrame, TeeReader, Viewport,
};

const SH_COEFS: [[f32; 4]; 16] = [
    [
        1.74370718002319336,
        1.74722564220428467,
        1.7776036262512207,
        1.18455326557159424,
    ],
    [
        0.0285536199808120728,
        0.694771349430084229,
        0.03411903977394104,
        3.4746591381917824e-08,
    ],
    [
        0.0432392396032810211,
        0.0440542176365852356,
        0.778257608413696289,
        1.96939016205988082e-08,
    ],
    [
        0.627923309803009033,
        0.0201491285115480423,
        0.0462785884737968445,
        -6.05772783046631957e-08,
    ],
    [
        0.0271544735878705978,
        0.0186634529381990433,
        0.00635719392448663712,
        0.0508591160178184509,
    ],
    [
        0.0160119365900754929,
        0.0159059464931488037,
        0.0334366597235202789,
        0.0550790876150131226,
    ],
    [
        0.0197454746812582016,
        0.0412799231708049774,
        -0.0124673694372177124,
        0.091043427586555481,
    ],
    [
        -0.0103152133524417877,
        -0.000563298584893345833,
        0.0180064737796783447,
        0.0197699647396802902,
    ],
    [
        -0.0228151362389326096,
        0.0242609847337007523,
        -0.0239147115498781204,
        -0.00754963979125022888,
    ],
    [
        -0.016609853133559227,
        0.00716178072616457939,
        -0.00636536208912730217,
        4.44060921367395167e-09,
    ],
    [
        -0.0107341120019555092,
        -0.016083618625998497,
        -0.0129809658974409103,
        -4.34890212730465464e-09,
    ],
    [
        -0.000943164690397679806,
        -0.0221798326820135117,
        -0.0277967583388090134,
        -2.54724756842961142e-08,
    ],
    [
        0.010390598326921463,
        0.00321987480856478214,
        -0.0316285006701946259,
        -4.48716219736411404e-09,
    ],
    [
        -0.0305650513619184494,
        0.000451165484264492989,
        -0.0123444544151425362,
        -1.76184151712277526e-08,
    ],
    [
        -0.00832935608923435211,
        0.00844020955264568329,
        0.00625753495842218399,
        6.42785069615570137e-09,
    ],
    [
        0.0123509559780359268,
        0.0113992318511009216,
        0.00409724144265055656,
        1.84686932414024341e-08,
    ],
];

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    let filename = opt.input.as_os_str().to_str().unwrap();

    let mut octree: Octree<f32, f32> = {
        let in_file = File::open(filename).unwrap();

        let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

        let mut buf = BufReader::new(in_file);

        pb.message(&format!("decoding {}: ", filename));

        pb.set_units(pbr::Units::Bytes);
        let mut tee = TeeReader::new(&mut buf, &mut pb);

        let octree: Octree<f64, u8> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!");

        octree.into()
    };

    for octant in octree.borrow_mut().into_iter() {
        let pc: &PointCloud<f32, f32> = octant.points().into();
        let mean = pc.mean();
        let mut coefs = [Vector4::<f32>::zeros(); 121];
        // coefs[0] = mean.color / 0.28209478;
        for i in 0..SH_COEFS.len() {
            coefs[i] = SH_COEFS[i].into();
        }
        octant.sh_approximation = Some(SHVertex::new(mean.position, coefs.into()));
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

    println!("using device {}", physical_device.properties().device_name);
    let (device, mut queues) = Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),

            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            enabled_features: Features {
                buffer_device_address: true,
                shader_int64: true,
                ..Features::none()
            },
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

    let window_size = surface.window().inner_size();
    let aspect_ratio = window_size.width as f32 / window_size.height as f32;
    let mut camera = PerspectiveCamera::new(
        Point3::new(0.0, 19.337029, -47.544266),
        Vector3::new(0.4617872, 0.0, 0.0),
        aspect_ratio,
    );

    let renderer = Arc::new(OctreeRenderer::new(
        device.clone(),
        queue.clone(),
        scene_subpass,
        viewport.clone(),
        octree.clone(),
        &camera,
    ));

    renderer.set_point_size(1);
    renderer.frustum_culling();

    let renderer_clone = renderer.clone();

    let (tx, rx) = mpsc::channel();
    let frustum_culling_thread = thread::spawn(move || loop {
        renderer_clone.frustum_culling();
        match rx.try_recv() {
            Ok(_) | Err(TryRecvError::Disconnected) => {
                break;
            }
            Err(TryRecvError::Empty) => {}
        }
    });

    let mut camera_controller = CameraController::new(50., 1.);

    let mut last_update_inst = Instant::now();

    let mut last_mouse_position: Option<PhysicalPosition<f64>> = None;
    let mut mouse_pressed = false;

    event_loop.run_return(move |event, _, control_flow| match event {
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
        Event::WindowEvent {
            event:
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(key),
                            state,
                            ..
                        },
                    ..
                },
            ..
        } => {
            if state == ElementState::Released && key == VirtualKeyCode::C {
                println!(
                    "camera: pos: {:?}, rot: {:?}",
                    camera.position(),
                    camera.rotation()
                )
            }
            camera_controller.process_keyboard(key, state);
        }
        // Event::DeviceEvent { event, .. } => match event {
        //     DeviceEvent::Key(KeyboardInput {
        //         virtual_keycode: Some(key),
        //         state,
        //         ..
        //     }) => {
        //         camera_controller.process_keyboard(key, state);
        //     }
        //     _ => {}
        // },
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

            let pc_cb = renderer.render();
            frame.render(queue.clone(), pc_cb.clone());
        }
        Event::LoopDestroyed => {
            let _ = tx.send(());
        }
        _ => (),
    });

    frustum_culling_thread.join().unwrap();
}
