use std::{f32::consts::FRAC_1_PI, sync::Arc};

use camera::Projection;
use image::{ImageBuffer, Rgba};
use nalgebra::{Point3, Vector2, Vector3};
use sh::{calc_sh_fixed, to_spherical};
use tch::Kind;
use vertex::NUM_COEFS;
use vulkano::{
    buffer::{BufferContents, BufferUsage, CpuAccessibleBuffer},
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
mod io;
mod octree;
mod ply;
mod pointcloud;
mod renderer;
pub mod sh;
mod tee;
mod vertex;

pub use avg_color::ImageAvgColor;
pub use camera::{
    Camera, CameraController, OrthographicCamera, OrthographicProjection, PerspectiveCamera,
    PerspectiveProjection,
};
pub use io::{
    export_ply, load_cameras, load_octree_with_progress_bar, load_raw_coefs,
    save_octree_with_progress_bar,
};
pub use octree::{LeafNode, Node, Octree, OctreeIterator};
pub use pointcloud::{CubeBoundingBox, PointCloud, PointCloudGPU};
pub use renderer::{
    Frame, LoDMode, OctreeRenderer, PointCloudRenderer, RenderMode, SurfaceFrame, Viewport,
};
pub use tee::{TeeReader, TeeWriter};
pub use vertex::{BaseFloat, SHCoefficients, SHVertex, Vertex};

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
                format: vulkano::format::Format::D32_SFLOAT,
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
    pub point_size: u32,
    pub background_color: [f32; 4],
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            point_size: 1,
            background_color: [0.; 4],
        }
    }
}

pub struct OfflineRenderer {
    renderer: PointCloudRenderer,
    frame: Frame,
    queue: Arc<Queue>,
    calc_img_average: ImageAvgColor,
    img_size: u32,

    cpu_buffer: Option<Arc<CpuAccessibleBuffer<[u8]>>>,
}

impl OfflineRenderer {
    pub fn new(img_size: u32, render_settings: RenderSettings, save_renders: bool) -> Self {
        let instance = Instance::new(InstanceCreateInfo::default()).unwrap();

        let device_extensions = DeviceExtensions::none();

        let (physical_device, queue_family) =
            select_physical_device(&instance, &device_extensions, None);

        println!("using device {}", physical_device.properties().device_name);

        let (device, mut queues) = Device::new(
            // Which physical device to connect to.
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,

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

        let cpu_buffer = if save_renders {
            Some(
                CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::transfer_dst(),
                    false,
                    (0..img_size * img_size * 4).map(|_| 0u8),
                )
                .unwrap(),
            )
        } else {
            None
        };

        renderer.set_point_size(render_settings.point_size);
        frame.set_background(render_settings.background_color);

        let src_img_view = ImageView::new_default(frame.buffer.image().clone()).unwrap();

        let calc_img_average =
            ImageAvgColor::new(device.clone(), queue.clone(), src_img_view, img_size);

        OfflineRenderer {
            renderer,
            frame,
            queue,
            calc_img_average,
            img_size,
            cpu_buffer,
        }
    }

    pub fn render(&mut self, camera: Camera<impl Projection>, pc: &PointCloudGPU) -> Rgba<u8> {
        self.renderer.set_camera(&camera);
        let cb = self.renderer.render_point_cloud(self.queue.clone(), pc);

        let cb_render = self
            .frame
            .render(self.queue.clone(), cb, self.cpu_buffer.clone());

        let avg_color = self.calc_img_average.calc_average_color(cb_render);
        return avg_color;
    }

    pub fn last_image(&self) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
        if let Some(buffer) = &self.cpu_buffer {
            let img_data = buffer.read().unwrap();
            ImageBuffer::from_raw(self.img_size, self.img_size, img_data[..].to_vec()).unwrap()
        } else {
            panic!("save_renders is false so the image is not stored");
        }
    }
}

unsafe impl DeviceOwned for OfflineRenderer {
    fn device(&self) -> &Arc<Device> {
        &self.queue.device()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoD {
    SHRep,
    Full,
}

pub fn merge_shs(
    shs: [Option<SHCoefficients>; 8],
    sample_locations: &Vec<Point3<f32>>,
) -> SHCoefficients {
    if !shs.iter().any(|s| s.is_some()) {
        return shs.iter().find(|s| s.is_some()).unwrap().unwrap();
    }

    let normals = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| {
        let z = i / 4;
        let y = (i - 4 * z) / 2;
        let x = i % 2;

        let sf = 2. / (3f32).sqrt();
        Vector3::new(x as f32 - 0.5f32, y as f32 - 0.5f32, z as f32 - 0.5f32) * sf
    });
    let sph_pos: Vec<Vector2<f32>> = sample_locations.iter().map(|p| to_spherical(p)).collect();
    let y = calc_sh_fixed::<NUM_COEFS>(sph_pos);

    let new_colors: Vec<Vector3<f32>> = sample_locations
        .iter()
        .enumerate()
        .map(|(p_i, p)| {
            let mut w_sum = 0.;
            let mut color = Vector3::zeros();

            for (i, sh_rep) in shs.iter().enumerate() {
                if let Some(sh_coefficients) = sh_rep {
                    let n = normals[i];
                    let w = 1. - FRAC_1_PI * n.angle(&p.coords);
                    let mut a_color = Vector3::zeros();
                    for y_i in 0..NUM_COEFS {
                        a_color += y[p_i][y_i] * sh_coefficients.0[y_i];
                    }
                    color += w * a_color;
                    w_sum += w;
                }
            }
            color / w_sum
        })
        .collect();

    // calculate the new coefficients using torches
    // least squares solver

    let device = tch::Device::Cpu;

    let y_t = tch::Tensor::of_data_size(
        y.as_bytes(),
        &[y.len() as i64, NUM_COEFS as i64],
        Kind::Float,
    )
    .to(device);
    let target = tch::Tensor::of_data_size(
        new_colors.as_bytes(),
        &[new_colors.len() as i64, 3],
        Kind::Float,
    )
    .to(device);

    let a = y_t.transpose(0, 1).matmul(&y_t);
    let b = y_t.transpose(0, 1).matmul(&target);

    let solution = a.linalg_lstsq(&b, None, "gelsy").0;

    let mut new_coefs = SHCoefficients([Vector3::zeros(); NUM_COEFS]);
    for (i, v) in Vec::<f32>::from(solution).iter().enumerate() {
        new_coefs.0[i / 3][i % 3] = *v;
    }

    new_coefs
}
