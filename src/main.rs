use std::sync::Arc;

use frame::Frame;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily};
use vulkano::device::{Device, DeviceExtensions, Features};
use vulkano::format::Format;
use vulkano::instance::Instance;
use vulkano::render_pass::{RenderPass, Subpass};
use vulkano::swapchain::Surface;
use vulkano::Version;
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

mod frame;
mod pc_render;
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
                .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
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

fn get_render_pass(device: Arc<Device>, swapchain_format: Format) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain_format,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {}
        }
    )
    .unwrap()
}

fn main() {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, Version::V1_1, &required_extensions, None)
        .expect("failed to create instance");

    let event_loop = EventLoop::new(); // ignore this for now
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) =
        select_physical_device(&instance, surface.clone(), &device_extensions);

    let (device, mut queues) = {
        Device::new(
            physical_device,
            &Features::none(),
            &physical_device
                .required_extensions()
                .union(&device_extensions), // new
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };

    let queue = queues.next().unwrap();

    let caps = surface
        .capabilities(physical_device)
        .expect("failed to get surface capabilities");

    let swapchain_format = caps.supported_formats[0].0;

    let render_pass = get_render_pass(device.clone(), swapchain_format);

    let mut frame = Frame::new(
        surface.clone(),
        device.clone(),
        physical_device.clone(),
        queue.clone(),
        render_pass.clone(),
        swapchain_format,
        render_pass.clone(),
    );

    let point_cloud_subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let renderer = pc_render::PointCloudRenderer::new(queue.clone(), point_cloud_subpass);

    let mut window_resized = false;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            window_resized = true;
            surface.window().request_redraw();
        }
        Event::MainEventsCleared => {}

        Event::RedrawEventsCleared => {
            if window_resized {
                frame.recreate(window_resized.then(|| surface.window().inner_size()));
            }
        }
        Event::RedrawRequested(..) => {
            let cb = renderer.draw([800, 600]);
            frame.next_frame(queue.clone(), cb);
        }
        _ => (),
    });
}
