use image::{ImageBuffer, Rgba};
use nalgebra::Point3;
use std::path::PathBuf;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BlitImageInfo, CommandBufferUsage, CopyImageToBufferInfo,
};
use vulkano::device::{Features, Queue};
use vulkano::format::Format;
use vulkano::image::{ImageDimensions, StorageImage};
use vulkano::sync::{self, GpuFuture};

use punctum::{
    get_render_pass, load_octree_with_progress_bar, select_physical_device, Frame, LoDMode, Octree,
    OctreeRenderer, PerspectiveCamera, RenderMode, Viewport,
};
use structopt::StructOpt;
use vulkano::device::DeviceExtensions;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::render_pass::{RenderPass, Subpass};

#[derive(StructOpt, Debug)]
#[structopt(name = "Offline Renderer")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "output_file", parse(from_os_str))]
    output_file: PathBuf,

    // image/render width
    #[structopt(long, short, default_value = "256")]
    width: u32,

    // image/render height
    #[structopt(long, short, default_value = "256")]
    height: u32,

    /// LoD threshold in pixels
    #[structopt(long, default_value = "1")]
    lod_threshold: u32,

    /// camera x position
    #[structopt(long, short, default_value = "0.")]
    x_camera: f32,
    /// camera y position
    #[structopt(long, short, default_value = "0.")]
    y_camera: f32,
    /// camera z position
    #[structopt(long, short, default_value = "0.")]
    z_camera: f32,
}

fn render_from_viewpoints(
    output_file: PathBuf,
    device: Arc<Device>,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    image_format: Format,
    render_size: [u32; 2],
    culling_mode: LoDMode,
    octree: Arc<Octree<f32>>,
    camera: PerspectiveCamera,
    lod_threshold: u32,
) {
    let viewport = Viewport::new(render_size);

    let mut camera = camera;

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
        &camera,
    ));
    renderer.set_point_size(1);

    let target_buffer: Arc<CpuAccessibleBuffer<[u8]>> = unsafe {
        CpuAccessibleBuffer::<[u8]>::uninitialized_array(
            device.clone(),
            (render_size[0] * render_size[1]) as u64 * image_format.block_size().unwrap(),
            BufferUsage::transfer_dst(),
            false,
        )
        .unwrap()
    };

    let target_image: Arc<StorageImage> = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: render_size[0],
            height: render_size[1],
            array_layers: 1,
        },
        image_format,
        [queue.family()],
    )
    .unwrap();

    renderer.set_highlight_sh(false);

    camera.adjust_znear_zfar(octree.bbox());

    renderer.set_camera(&camera);
    renderer.set_viewport(Viewport::new(render_size));
    renderer.update_uniforms();
    renderer.update_lod(culling_mode, lod_threshold);

    let pc_cb = renderer.render(RenderMode::Both, false);

    let future = frame.render(queue.clone(), pc_cb.clone(), None);

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let mut blit_info = BlitImageInfo::images(frame.buffer.image().clone(), target_image.clone());
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
    let image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(render_size[0], render_size[1], buf.to_vec()).unwrap();
    image.save(output_file).unwrap()
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

    let image_size = [opt.width, opt.height];

    let camera =
        PerspectiveCamera::look_at_origin(Point3::new(opt.x_camera, opt.y_camera, opt.z_camera));

    render_from_viewpoints(
        opt.output_file.clone(),
        device.clone(),
        queue.clone(),
        render_pass.clone(),
        image_format,
        image_size,
        LoDMode::Mixed,
        octree.clone(),
        camera,
        opt.lod_threshold,
    );
}
