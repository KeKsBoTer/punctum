use image::{ImageBuffer, Rgba};
use nalgebra::Point3;
use pbr::{MultiBar, Pipe, ProgressBar};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::device::{Features, Queue};
use vulkano::format::Format;
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::sync::{self, GpuFuture};

use punctum::{
    get_render_pass, load_cameras, load_octree_with_progress_bar, select_physical_device, Frame,
    LoDMode, Octree, OctreeRenderer, PerspectiveCamera, PerspectiveProjection, RenderMode,
    Viewport,
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

    #[structopt(name = "output_folder", parse(from_os_str))]
    output_folder: PathBuf,

    #[structopt(long, default_value = "256")]
    img_size: u32,

    #[structopt(long)]
    parallel: bool,
}

fn render_from_viewpoints(
    name: String,
    output_folder: PathBuf,
    device: Arc<Device>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    image_format: Format,
    render_size: [u32; 2],
    target_img_size: [u32; 2],
    culling_mode: LoDMode,
    highlight_sh: bool,
    octree: Arc<Octree<f32>>,
    cameras: Vec<PerspectiveCamera>,
    mut pbr: ProgressBar<Pipe>,
    parallel: bool,
) -> Option<JoinHandle<()>> {
    let mut f = move || {
        pbr.message(name.as_str());

        let viewport = Viewport::new(render_size);

        let mut frame = Frame::new(
            device.clone(),
            render_pass.clone(),
            image_format,
            render_size,
        );
        frame.set_background([0., 0., 0., 0.]);

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
                BufferUsage::transfer_dst(),
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

        let mut images = Vec::with_capacity(cameras.len());

        renderer.set_highlight_sh(highlight_sh);

        for (i, camera) in cameras.iter().enumerate() {
            let mut camera = camera.clone();
            let c_pos = camera.view().transform_point(&Point3::origin());
            camera.translate(c_pos.coords * 50. * 3f32.sqrt());
            camera.adjust_znear_zfar(octree.bbox());

            renderer.set_camera(&camera);
            renderer.update_uniforms();
            renderer.update_lod(culling_mode, 1. / render_size[1] as f32);

            let pc_cb = renderer.render(RenderMode::Both, false);

            let future = frame.render(queue.clone(), pc_cb.clone(), None);

            let mut builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            let mut blit_info =
                BlitImageInfo::images(frame.buffer.image().clone(), target_image.clone());
            blit_info.filter = vulkano::sampler::Filter::Linear;

            builder.blit_image(blit_info).unwrap();
            builder
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                    target_image.clone(),
                    target_buffer.clone(),
                ))
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
            images.push((output_folder.join(format!("{}_{}.png", name, i)), image));

            pbr.inc();
        }
        // do image saving in parallel
        images
            .par_iter()
            .map(|(filename, image)| image.save(filename).unwrap())
            .count();
        pbr.finish()
    };

    if parallel {
        Some(thread::spawn(f))
    } else {
        f();
        None
    }
}

fn main() {
    let opt = Opt::from_args();
    let filename = opt.input;

    let octree: Octree<f32> = load_octree_with_progress_bar(&filename).unwrap().into();
    let octree = Arc::new(octree);

    let instance = Instance::new(InstanceCreateInfo::application_from_cargo_toml()).unwrap();

    let device_extensions = DeviceExtensions::none();

    let (physical_device, queue_family) =
        select_physical_device(&instance, &device_extensions, None);

    let (device, mut queues) = Device::new(
        // Which physical device to connect to.
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,

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

    if !opt.output_folder.exists() {
        std::fs::create_dir(opt.output_folder.clone()).unwrap();
    }

    let mb = MultiBar::new();
    mb.println("Rendering:");

    let p_render_sh = mb.create_bar(cameras.len() as u64);
    let p_render_no_sh = mb.create_bar(cameras.len() as u64);
    let p_sh_mask = mb.create_bar(cameras.len() as u64);
    let p_multi_sampling = mb.create_bar(cameras.len() as u64);

    let pb_thread = thread::spawn(move || mb.listen());

    let render_sh = render_from_viewpoints(
        "render_sh".to_string(),
        opt.output_folder.clone(),
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        image_format,
        image_size,
        image_size,
        LoDMode::Mixed,
        false,
        octree.clone(),
        cameras.clone(),
        p_render_sh,
        opt.parallel,
    );
    let render_no_sh = render_from_viewpoints(
        "render_no_sh".to_string(),
        opt.output_folder.clone(),
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        image_format,
        image_size,
        image_size,
        LoDMode::OctantsOnly,
        false,
        octree.clone(),
        cameras.clone(),
        p_render_no_sh,
        opt.parallel,
    );
    let sh_mask = render_from_viewpoints(
        "sh_mask".to_string(),
        opt.output_folder.clone(),
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        image_format,
        image_size,
        image_size,
        LoDMode::Mixed,
        true,
        octree.clone(),
        cameras.clone(),
        p_sh_mask,
        opt.parallel,
    );

    let multi_sampling = render_from_viewpoints(
        "multisampled".to_string(),
        opt.output_folder.clone(),
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        image_format,
        [image_size[0] * 4, image_size[1] * 4],
        image_size,
        LoDMode::Mixed,
        false,
        octree.clone(),
        cameras.clone(),
        p_multi_sampling,
        opt.parallel,
    );

    if let Some(t) = render_sh {
        t.join().unwrap();
    }
    if let Some(t) = render_no_sh {
        t.join().unwrap();
    }
    if let Some(t) = sh_mask {
        t.join().unwrap();
    }
    if let Some(t) = multi_sampling {
        t.join().unwrap();
    }
    pb_thread.join().unwrap();
}
