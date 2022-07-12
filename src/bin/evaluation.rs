use image::{ImageBuffer, Rgba};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::device::{Features, Queue};
use vulkano::format::Format;
use vulkano::half::f16;
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::sync::{self, GpuFuture};

use punctum::{
    get_render_pass, load_cameras, load_octree_with_progress_bar, select_physical_device, Frame,
    Octree, OctreeRenderer, PerspectiveCamera, PerspectiveProjection, Viewport,
};
use structopt::StructOpt;
use vulkano::device::DeviceExtensions;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::render_pass::{RenderPass, Subpass};

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,

    #[structopt(long, default_value = "256")]
    img_size: u32,
}

fn render_from_viewpoints(
    name: String,
    device: Arc<Device>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    image_format: Format,
    render_size: [u32; 2],
    target_img_size: [u32; 2],
    render_sh: bool,
    highlight_sh: bool,
    octree: Arc<Octree<f32, f32>>,
    cameras: Vec<PerspectiveCamera>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let viewport = Viewport::new(render_size);

        let frame = Frame::new(
            device.clone(),
            render_pass.clone(),
            image_format,
            render_size,
        );

        let scene_subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let renderer = Arc::new(OctreeRenderer::new(
            device.clone(),
            queue.clone(),
            scene_subpass.clone(),
            viewport.clone(),
            octree.clone(),
            &cameras[0],
        ));
        renderer.set_point_size(1);

        let target_buffer: Arc<CpuAccessibleBuffer<[u8]>> = unsafe {
            CpuAccessibleBuffer::<[u8]>::uninitialized_array(
                device.clone(),
                (target_img_size[0] * target_img_size[1]) as u64
                    * image_format.block_size().unwrap(),
                BufferUsage::transfer_destination(),
                false,
            )
            .unwrap()
        };

        let target_image: Arc<StorageImage> = StorageImage::new(
            device.clone(),
            ImageDimensions::Dim2d {
                width: target_img_size[0],
                height: target_img_size[1],
                array_layers: 1,
            },
            image_format,
            [queue.family()],
        )
        .unwrap();

        let depth_img = StorageImage::new(
            device.clone(),
            ImageDimensions::Dim2d {
                width: render_size[0],
                height: render_size[1],
                array_layers: 1,
            },
            Format::R16_SFLOAT,
            [queue.family()],
        )
        .unwrap();

        let depth_buffer: Arc<CpuAccessibleBuffer<[f16]>> = unsafe {
            CpuAccessibleBuffer::uninitialized_array(
                device.clone(),
                (target_img_size[0] * target_img_size[1]) as u64 * 2,
                BufferUsage::transfer_destination(),
                false,
            )
            .unwrap()
        };

        let mut images = Vec::with_capacity(cameras.len());

        renderer.set_highlight_sh(highlight_sh);

        for (i, camera) in cameras.iter().enumerate() {
            let mut camera = camera.clone();
            camera.translate(camera.position().coords * 100. * 3f32.sqrt());
            camera.adjust_znear_zfar(octree.bbox());

            renderer.set_camera(&camera);
            renderer.update_uniforms();
            renderer.frustum_culling(render_sh);
            let pc_cb = renderer.render(true, true);

            let future = frame.render(queue.clone(), pc_cb.clone(), None);

            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            builder
                .blit_image(
                    frame.buffer.image().clone(),
                    [0, 0, 0],
                    [render_size[0] as i32, render_size[1] as i32, 1],
                    0,
                    0,
                    target_image.clone(),
                    [0, 0, 0],
                    [target_img_size[0] as i32, target_img_size[1] as i32, 1],
                    0,
                    0,
                    1,
                    vulkano::sampler::Filter::Linear,
                )
                .unwrap();
            builder
                .copy_image_to_buffer(target_image.clone(), target_buffer.clone())
                .unwrap();

            builder
                .blit_image(
                    frame.buffer.depth_buffer().clone(),
                    [0, 0, 0],
                    [render_size[0] as i32, render_size[1] as i32, 1],
                    0,
                    0,
                    depth_img.clone(),
                    [0, 0, 0],
                    [render_size[0] as i32, render_size[1] as i32, 1],
                    0,
                    0,
                    1,
                    vulkano::sampler::Filter::Nearest,
                )
                .unwrap();

            let cb = builder.build().unwrap();

            sync::now(device.clone())
                .join(future)
                .then_execute_same_queue(cb)
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap()
                .wait(None)
                .unwrap();

            let buffer_content = target_buffer.read().unwrap();
            let buf = &buffer_content[..];
            let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
                target_img_size[0],
                target_img_size[1],
                buf.to_vec(),
            )
            .unwrap();
            images.push((format!("renders/{}_{}.png", name, i), image));

            let buffer_content = depth_buffer.read().unwrap();
            let buf: Vec<u8> = buffer_content[..]
                .iter()
                .flat_map(|v| {
                    let a = (v.to_f32() * 255.) as u8;
                    vec![a, a, a, a]
                })
                .collect();
            let image =
                ImageBuffer::<Rgba<u8>, _>::from_vec(target_img_size[0], target_img_size[1], buf)
                    .unwrap();
            images.push((format!("renders/{}_{}_d.png", name, i), image));
        }
        // do image saving in parallel
        images
            .par_iter()
            .map(|(filename, image)| image.save(filename).unwrap())
            .count();
    })
}

fn main() {
    let opt = Opt::from_args();
    let filename = opt.input;

    let octree: Octree<f32, f32> = load_octree_with_progress_bar(&filename).unwrap().into();
    let octree = Arc::new(octree);

    let instance = Instance::new(InstanceCreateInfo::application_from_cargo_toml()).unwrap();

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
            enabled_features: Features {
                shader_int64: true,
                ..Features::none()
            },
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let image_format = vulkano::format::Format::R8G8B8A8_UNORM;

    let render_pass = get_render_pass(device.clone(), image_format);

    let image_size = [opt.img_size, opt.img_size];

    let aspect_ratio = image_size[0] as f32 / image_size[1] as f32;

    let cameras = load_cameras(
        "sphere.ply",
        PerspectiveProjection {
            fovy: PI / 2.,
            aspect_ratio: aspect_ratio,
        },
    )
    .unwrap();

    let render_sh = render_from_viewpoints(
        "render_sh".to_string(),
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        image_format,
        image_size,
        image_size,
        true,
        false,
        octree.clone(),
        cameras.clone(),
    );
    let render_no_sh = render_from_viewpoints(
        "render_no_sh".to_string(),
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        image_format,
        image_size,
        image_size,
        false,
        false,
        octree.clone(),
        cameras.clone(),
    );
    let sh_mask = render_from_viewpoints(
        "sh_mask".to_string(),
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        image_format,
        image_size,
        image_size,
        true,
        true,
        octree.clone(),
        cameras.clone(),
    );
    let multi_sampling = render_from_viewpoints(
        "multisampled".to_string(),
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        image_format,
        [image_size[0] * 4, image_size[1] * 4],
        image_size,
        false,
        false,
        octree.clone(),
        cameras.clone(),
    );

    render_sh.join().unwrap();
    render_no_sh.join().unwrap();
    sh_mask.join().unwrap();
    multi_sampling.join().unwrap();
}
