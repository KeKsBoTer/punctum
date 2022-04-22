mod camera;
mod pointcloud;
mod renderer;
mod vertex;

use std::sync::Arc;

pub use camera::{Camera, CameraController};
use cgmath::{vec4, EuclideanSpace, Point3, Vector4};
use image::Rgba;
use ply_rs::ply::{self, Addable, ElementDef, PropertyDef, PropertyType, ScalarType};
pub use pointcloud::{PointCloud, PointCloudGPU};
use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use renderer::Frame;
pub use renderer::{PointCloudRenderer, SurfaceFrame, Viewport};
pub use vertex::Vertex;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType, QueueFamily},
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo,
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

    buffer: Arc<CpuAccessibleBuffer<[[u8; 4]]>>,
}

impl OfflineRenderer {
    pub fn new(pc: Arc<PointCloud>, img_size: u32, render_settings: RenderSettings) -> Self {
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

        let pc_gpu = PointCloudGPU::from_point_cloud(device.clone(), pc.clone());

        // let pixel_format_size = image_format.components()

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
            pc: pc_gpu,
        }
    }

    pub fn render(&mut self, camera: Camera) -> Rgba<u8> {
        self.renderer.set_camera(&camera);
        let cb = self
            .renderer
            .render_point_cloud(self.queue.clone(), &self.pc);

        self.frame
            .render(self.queue.clone(), cb, self.buffer.clone());

        let buffer_content = self.buffer.read().unwrap();

        calc_average_color(&buffer_content)
    }
}

fn calc_average_color(data: &[[u8; 4]]) -> Rgba<u8> {
    let start = vec4(0., 0., 0., 0.);
    let mean = data
        .par_chunks_exact(1024)
        .fold(
            || start,
            |acc, chunk| {
                acc + chunk.iter().fold(start, |acc, item| {
                    acc + Vector4::from(item.map(|v| v as f32)) / (255. * 255.) * (item[3] as f32)
                })
            },
        )
        .reduce(|| start, |acc, item| acc + item);
    Rgba(mean.map(|v| (v * 255. / mean[3]).round() as u8).into())
}

#[derive(Clone, Copy)]
pub struct PerceivedColor {
    pub pos: Point3<f32>,
    pub color: Rgba<u8>,
}

impl PerceivedColor {
    pub fn element_def(name: String) -> ElementDef {
        let mut point_element = ElementDef::new(name);
        let p = PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("red".to_string(), PropertyType::Scalar(ScalarType::UChar));
        point_element.properties.add(p);
        let p = PropertyDef::new("green".to_string(), PropertyType::Scalar(ScalarType::UChar));
        point_element.properties.add(p);
        let p = PropertyDef::new("blue".to_string(), PropertyType::Scalar(ScalarType::UChar));
        point_element.properties.add(p);
        let p = PropertyDef::new("alpha".to_string(), PropertyType::Scalar(ScalarType::UChar));
        point_element.properties.add(p);
        return point_element;
    }
}

impl ply::PropertyAccess for PerceivedColor {
    fn new() -> Self {
        PerceivedColor {
            pos: Point3::origin(),
            color: Rgba([0; 4]),
        }
    }

    fn set_property(&mut self, key: String, property: ply::Property) {
        match (key.as_ref(), property) {
            ("x", ply::Property::Float(v)) => self.pos[0] = v,
            ("y", ply::Property::Float(v)) => self.pos[1] = v,
            ("z", ply::Property::Float(v)) => self.pos[2] = v,
            ("red", ply::Property::UChar(v)) => self.color[0] = v,
            ("green", ply::Property::UChar(v)) => self.color[1] = v,
            ("blue", ply::Property::UChar(v)) => self.color[2] = v,
            ("alpha", ply::Property::UChar(v)) => self.color[3] = v,
            ("nx", ply::Property::Float(_)) => {}
            ("ny", ply::Property::Float(_)) => {}
            ("nz", ply::Property::Float(_)) => {}
            ("vertex_indices", _) => {}
            (k, _) => panic!("Vertex: Unexpected key/value combination: key: {}", k),
        }
    }
    fn get_uchar(&self, _property_name: &String) -> Option<u8> {
        match _property_name.as_str() {
            "red" => Some(self.color[0]),
            "green" => Some(self.color[1]),
            "blue" => Some(self.color[2]),
            "alpha" => Some(self.color[3]),
            _ => None,
        }
    }
    fn get_float(&self, _property_name: &String) -> Option<f32> {
        match _property_name.as_str() {
            "x" => Some(self.pos.x),
            "y" => Some(self.pos.y),
            "z" => Some(self.pos.z),
            _ => None,
        }
    }
}
