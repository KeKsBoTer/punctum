use std::{fs::File, io::BufWriter, path::PathBuf, sync::Arc};

use camera::Projection;
use image::Rgba;
use nalgebra::{vector, Vector4};
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned, Queue, QueueCreateInfo,
    },
    image::view::ImageView,
    instance::{Instance, InstanceCreateInfo},
    render_pass::{RenderPass, Subpass},
    swapchain::Surface,
};
use winit::window::Window;

mod avg_color;
mod camera;
mod octree;
mod pointcloud;
mod renderer;
mod tee;
mod vertex;

pub use avg_color::ImageAvgColor;
pub use camera::{Camera, CameraController, OrthographicCamera, PerspectiveCamera};
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
    calc_img_average: ImageAvgColor,

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

        let mut renderer =
            PointCloudRenderer::new(device.clone(), queue.clone(), scene_subpass, viewport);

        let target_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::transfer_destination(),
            false,
            (0..img_size * img_size).map(|_| [0u8; 4]),
        )
        .expect("failed to create buffer");

        renderer.set_point_size(render_settings.point_size);
        frame.set_background(render_settings.background_color);

        let src_img_view = ImageView::new_default(frame.buffer.image().clone()).unwrap();

        let calc_img_average =
            ImageAvgColor::new(device.clone(), queue.clone(), src_img_view, img_size);

        OfflineRenderer {
            renderer,
            frame,
            queue: queue,
            calc_img_average: calc_img_average,
            buffer: target_buffer,
        }
    }

    pub fn render(&mut self, camera: Camera<impl Projection>, pc: &PointCloudGPU) -> Rgba<u8> {
        self.renderer.set_camera(&camera);
        let cb = self.renderer.render_point_cloud(self.queue.clone(), pc);

        let cb_render = self
            .frame
            .render(self.queue.clone(), cb, self.buffer.clone());

        let avg_color = self.calc_img_average.calc_average_color(cb_render);
        return avg_color;
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
                    let rgba: Vector4<f32> = Vector4::from(*item).cast();
                    let a = rgba.w / 255.;
                    let rgb = rgba.xyz() / 255. * a;
                    acc + Vector4::new(rgb.x, rgb.y, rgb.z, a)
                })
            },
        )
        .reduce(|| start, |acc, item| acc + item);
    Rgba(
        mean.map(|v| (v * 255. / (data.len() as f32)).round() as u8)
            .into(),
    )
}

pub fn export_ply(output_file: &PathBuf, pc: &PointCloud<f32, u8>) {
    let mut file = BufWriter::new(File::create(output_file).unwrap());

    let mut ply = Ply::<Vertex<f32, u8>>::new();
    let mut elm_def = Vertex::<f32, u8>::element_def("vertex".to_string());
    elm_def.count = pc.points().len();
    ply.header.encoding = Encoding::Ascii;
    ply.header.elements.add(elm_def.clone());

    let w = Writer::<Vertex<f32, u8>>::new();
    w.write_header(&mut file, &ply.header).unwrap();
    w.write_payload_of_element(
        &mut file,
        pc.points(),
        ply.header.elements.get("vertex").unwrap(),
        &ply.header,
    )
    .unwrap();
}
