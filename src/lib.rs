use std::sync::Arc;

use image::Rgba;
use nalgebra::{vector, Vector4};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Queue, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, ImageAccess, ImageDimensions, ImageViewAbstract, StorageImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    render_pass::{RenderPass, Subpass},
    swapchain::Surface,
    sync::{self, GpuFuture},
};
use winit::window::Window;

mod camera;
mod octree;
mod pointcloud;
mod renderer;
mod tee;
mod vertex;

pub use camera::{Camera, CameraController};
pub use octree::{Node, Octree};
pub use pointcloud::{BoundingBox, PointCloud, PointCloudGPU};
pub use renderer::{PointCloudRenderer, SurfaceFrame, Viewport};
pub use tee::{TeeReader, TeeWriter};
pub use vertex::Vertex;

use renderer::Frame;

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

#[derive(Debug, Clone)]
pub struct RenderSettings {
    pub point_size: f32,
    pub background_color: [f32; 4],
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            point_size: 10.0,
            background_color: [0.; 4],
        }
    }
}

pub struct OfflineRenderer {
    renderer: PointCloudRenderer,
    frame: Frame,
    queue: Arc<Queue>,

    buffer: Arc<CpuAccessibleBuffer<[[u8; 4]]>>,
}

impl OfflineRenderer {
    pub fn new(img_size: u32, render_settings: RenderSettings) -> Self {
        let required_extensions = vulkano_win::required_extensions();
        let instance = Instance::new(InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        })
        .unwrap();

        let device_extensions = DeviceExtensions::none();

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

        let target_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::transfer_destination(),
            false,
            (0..img_size * img_size).map(|_| [0u8; 4]),
        )
        .expect("failed to create buffer");

        renderer.set_point_size(render_settings.point_size);
        frame.set_background(render_settings.background_color);

        OfflineRenderer {
            renderer,
            frame,
            queue: queue,
            buffer: target_buffer,
        }
    }

    pub fn render(&mut self, camera: Camera, pc: &PointCloudGPU) -> Rgba<u8> {
        self.renderer.set_camera(&camera);
        let cb = self.renderer.render_point_cloud(self.queue.clone(), pc);

        self.frame
            .render(self.queue.clone(), cb, self.buffer.clone());

        let buffer_content = self.buffer.read().unwrap();

        let result = calc_average_color(&buffer_content);
        return result;
    }
}

unsafe impl DeviceOwned for OfflineRenderer {
    fn device(&self) -> &Arc<Device> {
        &self.queue.device()
    }
}

pub fn calc_average_color(data: &[[u8; 4]]) -> Rgba<u8> {
    let start = vector!(0., 0., 0., 0.);
    let mean = data
        .par_chunks_exact(1024)
        .fold(
            || start,
            |acc, chunk| {
                acc + chunk.iter().fold(start, |acc, item| {
                    acc + Vector4::from(*item).cast() / (255. * 255.) * (item[3] as f32)
                })
            },
        )
        .reduce(|| start, |acc, item| acc + item);
    Rgba(mean.map(|v| (v * 255. / mean[3]).round() as u8).into())
}

// mod cs {
//     vulkano_shaders::shader! {
//         ty: "compute",
//         path: "src/renderer/shaders/image_reduction.comp"
//     }
// }

// pub struct ImageAvgColor {
//     device: Arc<Device>,
//     queue: Arc<Queue>,
//     command_buffer: Arc<PrimaryAutoCommandBuffer>,
//     target_buffer: Arc<CpuAccessibleBuffer<[[u8; 4]]>>,
// }

// const IMG_SIZES: [u32; 11] = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1];

// impl ImageAvgColor {
//     pub fn new(
//         device: Arc<Device>,
//         queue: Arc<Queue>,
//         src_img: Arc<dyn ImageViewAbstract>,
//         start_size: u32,
//     ) -> Self {
//         assert!(IMG_SIZES.contains(&start_size));

//         let sizes: Vec<u32> = IMG_SIZES.into_iter().filter(|s| *s < start_size).collect();

//         let images: Vec<Arc<ImageView<StorageImage>>> = sizes
//             .iter()
//             .map(|s| {
//                 let img = StorageImage::new(
//                     device.clone(),
//                     ImageDimensions::Dim2d {
//                         width: *s,
//                         height: *s,
//                         array_layers: 1,
//                     },
//                     Format::R8G8B8A8_UNORM,
//                     Some(queue.family()),
//                 )
//                 .unwrap();

//                 ImageView::new_default(img).unwrap()
//             })
//             .collect();

//         let target_buffer = CpuAccessibleBuffer::from_iter(
//             device.clone(),
//             BufferUsage::transfer_destination(),
//             false,
//             (0..1).map(|_| [0u8; 4]),
//         )
//         .unwrap();

//         let shader = cs::load(device.clone()).unwrap();

//         let compute_pipeline = ComputePipeline::new(
//             device.clone(),
//             shader.entry_point("main").unwrap(),
//             &(),
//             None,
//             |_| {},
//         )
//         .unwrap();

//         let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();

//         let mut builder = AutoCommandBufferBuilder::primary(
//             device.clone(),
//             queue.family(),
//             CommandBufferUsage::MultipleSubmit,
//         )
//         .unwrap();

//         builder.bind_pipeline_compute(compute_pipeline.clone());

//         for i in 0..sizes.len() {
//             let set = PersistentDescriptorSet::new(
//                 layout.clone(),
//                 [
//                     if i == 0 {
//                         WriteDescriptorSet::image_view(0, src_img.clone())
//                     } else {
//                         WriteDescriptorSet::image_view(0, images[i - 1].clone())
//                     },
//                     WriteDescriptorSet::image_view(1, images[i].clone()),
//                 ],
//             )
//             .unwrap();
//             builder
//                 .bind_descriptor_sets(
//                     PipelineBindPoint::Compute,
//                     compute_pipeline.layout().clone(),
//                     0,
//                     set,
//                 )
//                 .dispatch([sizes[i], sizes[i], 1])
//                 .unwrap();
//         }

//         builder
//             .copy_image_to_buffer(
//                 images.last().unwrap().image().clone(),
//                 target_buffer.clone(),
//             )
//             .unwrap();

//         let command_buffer = Arc::new(builder.build().unwrap());

//         ImageAvgColor {
//             device,
//             queue,
//             command_buffer,
//             target_buffer,
//         }
//     }

//     fn calc_average_color(&self) -> Rgba<u8> {
//         let future = sync::now(self.device.clone())
//             .then_execute(self.queue.clone(), self.command_buffer.clone())
//             .unwrap()
//             .then_signal_fence_and_flush()
//             .unwrap();

//         future.wait(None).unwrap();

//         let buffer_content = self.target_buffer.read().unwrap();

//         return Rgba([0, 0, 0, 0]);
//     }
// }
