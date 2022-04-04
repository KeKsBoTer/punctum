mod camera;
mod pointcloud;
mod renderer;
mod vertex;

use std::sync::Arc;

pub use camera::{Camera, CameraController};
use image::Rgba;
pub use pointcloud::{PointCloud, PointCloudGPU};
use renderer::Frame;
pub use renderer::{PointCloudRenderer, SurfaceFrame, Viewport};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
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
    pc: PointCloudGPU,
    frame: Frame,
    queue: Arc<Queue>,

    compute_pipeline: Arc<ComputePipeline>,
    compute_ds: Arc<PersistentDescriptorSet>,

    buffer: Arc<CpuAccessibleBuffer<[u32; 4]>>,
}
mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/renderer/shaders/avg_color.comp"
    }
}

impl OfflineRenderer {
    pub fn new(pc: Arc<PointCloud>, img_size: u32, render_settings: RenderSettings) -> Self {
        let required_extensions = vulkano_win::required_extensions();
        let instance = Instance::new(InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        })
        .unwrap();

        let device_extensions = DeviceExtensions {
            ext_shader_atomic_float: true,
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

        let pc_gpu = PointCloudGPU::from_point_cloud(device.clone(), pc.clone());

        let pixel_format_size = image_format.block_size().unwrap() as u32;

        let target_buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::storage_buffer(),
            false,
            [0u32; 4],
        )
        .expect("failed to create buffer");

        renderer.set_point_size(render_settings.point_size);
        frame.set_background(render_settings.background_color);

        let shader = cs::load(device.clone()).expect("failed to create shader module");

        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .unwrap();

        let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, frame.buffer.image_view().clone()),
                WriteDescriptorSet::buffer(1, target_buffer.clone()),
            ],
        )
        .unwrap();

        OfflineRenderer {
            renderer,
            frame,
            queue: queue,
            buffer: target_buffer,
            pc: pc_gpu,
            compute_pipeline,
            compute_ds: set,
        }
    }

    pub fn render(&mut self, camera: Camera) -> Rgba<u8> {
        {
            let mut buf = self.buffer.write().unwrap();
            buf.fill(0)
        }

        self.renderer.set_camera(&camera);
        let cb = self
            .renderer
            .render_point_cloud(self.queue.clone(), &self.pc);

        let mut builder = AutoCommandBufferBuilder::primary(
            self.queue.device().clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                self.compute_ds.clone(),
            )
            .dispatch([256 / 32, 256 / 32, 1])
            .unwrap();

        let reduce_cb = Arc::new(builder.build().unwrap());

        self.frame.render(self.queue.clone(), cb, reduce_cb);

        let buffer_content = self.buffer.read().unwrap();
        println!("buffer: {:?}", buffer_content);

        Rgba([
            (buffer_content[0] / 16777216) as u8,
            (buffer_content[1] / 16777216) as u8,
            (buffer_content[2] / 16777216) as u8,
            (buffer_content[3] / 16777216) as u8,
        ])
    }
}
