use image::{ImageBuffer, Rgba};
use pbr::ProgressBar;
use std::f32::consts::PI;
use std::path::PathBuf;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::device::Features;
use vulkano::sync::{self, GpuFuture};

use punctum::{
    get_render_pass, load_cameras, load_octree_with_progress_bar, select_physical_device, Frame,
    Octree, OctreeRenderer, PerspectiveProjection, Viewport,
};
use structopt::StructOpt;
use vulkano::device::DeviceExtensions;
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::render_pass::Subpass;

#[derive(StructOpt, Debug)]
#[structopt(name = "Octree Builder")]
struct Opt {
    #[structopt(name = "input_octree", parse(from_os_str))]
    input: PathBuf,
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

    let image_size = [256, 256];

    let viewport = Viewport::new(image_size);

    let mut frame = Frame::new(
        device.clone(),
        render_pass.clone(),
        image_format,
        image_size,
    );

    let scene_subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    let aspect_ratio = image_size[0] as f32 / image_size[1] as f32;

    let cameras = load_cameras(
        "sphere.ply",
        PerspectiveProjection {
            fovy: PI / 2.,
            aspect_ratio: aspect_ratio,
        },
    )
    .unwrap();

    let renderer = Arc::new(OctreeRenderer::new(
        device.clone(),
        queue.clone(),
        scene_subpass.clone(),
        viewport.clone(),
        octree.clone(),
        &cameras[0],
    ));
    renderer.set_point_size(1);

    let target_images: Vec<Arc<CpuAccessibleBuffer<[u8]>>> = (0..cameras.len())
        .map(|_| unsafe {
            CpuAccessibleBuffer::<[u8]>::uninitialized_array(
                device.clone(),
                (image_size[0] * image_size[1]) as u64 * image_format.block_size().unwrap(),
                BufferUsage::transfer_destination(),
                false,
            )
            .unwrap()
        })
        .collect();

    let mut pb = ProgressBar::new(cameras.len() as u64);

    for (i, camera) in cameras.iter().enumerate() {
        let mut camera = camera.clone();
        camera.translate(camera.position().coords * 100. * 3f32.sqrt());
        camera.adjust_znear_zfar(octree.bbox());

        renderer.set_camera(&camera);
        renderer.update_uniforms();
        renderer.frustum_culling();
        let pc_cb = renderer.render(true, true);

        let target_img = &target_images[i];
        let future = frame.render(queue.clone(), pc_cb.clone(), Some(target_img.clone()));

        sync::now(device.clone())
            .join(future)
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let buffer_content = target_img.read().unwrap();
        let image =
            ImageBuffer::<Rgba<u8>, _>::from_raw(image_size[0], image_size[1], &buffer_content[..])
                .unwrap();
        image.save(format!("renders/render_{:}.png", i)).unwrap();
        pb.inc();
    }
}
