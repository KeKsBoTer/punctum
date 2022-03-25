mod camera;
mod pointcloud;
mod renderer;
mod vertex;

use std::sync::Arc;

pub use camera::{Camera, CameraController};
use image::{ImageBuffer, Rgba};
pub use pointcloud::{PointCloud, PointCloudGPU};
use renderer::Frame;
pub use renderer::{PointCloudRenderer, SurfaceFrame, Viewport};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo},
    render_pass::{RenderPass, Subpass},
    swapchain::Surface,
};
use winit::window::Window;

pub fn select_physical_device<'a>(
    instance: &'a Arc<Instance>,
    device_extensions: &DeviceExtensions,
    surface: Option<Arc<Surface<Window>>>,
) -> (PhysicalDevice<'a>, QueueFamily<'a>) {
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| {
                    let mut supports_surface = true;
                    if let Some(sur) = &surface {
                        supports_surface = q.supports_surface(&sur).unwrap_or(false)
                    }
                    return q.supports_graphics() && supports_surface;
                })
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

pub fn get_render_pass(
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

pub fn render_point_cloud(pc: Arc<PointCloud>, img_size: u32) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) =
        select_physical_device(&instance, &device_extensions, None);

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

    let image_format = vulkano::format::Format::R8G8B8A8_UNORM;

    println!("size: {:}", image_format.block_size().unwrap());

    let render_pass = get_render_pass(device.clone(), image_format);

    let viewport = Viewport::new([img_size, img_size]);

    let mut frame = Frame::new(
        device.clone(),
        render_pass.clone(),
        image_format,
        [img_size, img_size],
    );

    let scene_subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let mut renderer = PointCloudRenderer::new(device.clone(), scene_subpass, viewport);

    let camera = Camera::look_at_ortho(*pc.bounding_box());

    let pc_gpu = PointCloudGPU::from_point_cloud(device.clone(), pc.clone());

    let target_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::transfer_destination(),
        false,
        (0..img_size * img_size * 4).map(|_| 0u8),
    )
    .expect("failed to create buffer");

    renderer.set_camera(&camera);
    let cb = renderer.render_point_cloud(queue.clone(), &pc_gpu);
    frame.render(queue, cb, target_buffer.clone());

    let buffer_content = target_buffer.read().unwrap();

    ImageBuffer::<Rgba<u8>, _>::from_raw(img_size, img_size, buffer_content.to_vec()).unwrap()
}
