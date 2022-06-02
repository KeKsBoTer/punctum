use image::io::Reader as ImageReader;
use punctum::{calc_average_color, select_physical_device};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
    format::Format,
    image::{ImageDimensions, StorageImage},
    instance::{Instance, InstanceCreateInfo},
    sampler::Filter,
    sync::{self, GpuFuture},
};

fn main() {
    let instance = Instance::new(InstanceCreateInfo {
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

    let img = ImageReader::open("chrome1x.png").unwrap().decode().unwrap();
    let dimensions = ImageDimensions::Dim2d {
        width: img.width(),
        height: img.height(),
        array_layers: 1,
    };

    let buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::transfer_source(),
        false,
        img.to_rgba8()
            .pixels()
            .map(|p| p.0)
            .collect::<Vec<[u8; 4]>>(),
    )
    .unwrap();

    let target_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::transfer_destination(),
        false,
        (0..1).map(|_| [0u8; 4]),
    )
    .unwrap();

    let src_img = StorageImage::new(
        device.clone(),
        dimensions,
        Format::R8G8B8A8_UNORM,
        Some(queue.family()),
    )
    .unwrap();

    let target_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: 1,
            height: 1,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.family()),
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .copy_buffer_to_image(buffer.clone(), src_img.clone())
        .unwrap();

    builder
        .blit_image(
            src_img,
            [0, 0, 0],
            [255, 255, 1],
            0,
            0,
            target_image.clone(),
            [0, 0, 0],
            [1, 1, 1],
            0,
            0,
            1,
            Filter::Linear,
        )
        .unwrap();

    builder
        .copy_image_to_buffer(target_image, target_buffer.clone())
        .unwrap();
    // builder
    //     .copy_image_to_buffer_dimensions(texture, target_buffer, [0, 0, 0], [1, 1, 1], 0, 0, 0)
    //     .unwrap();

    let command_buffer = builder.build().unwrap();

    sync::now(device.clone())
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let src_buffer_content = buffer.read().unwrap();

    let baseline = calc_average_color(&src_buffer_content);

    let buffer_content = target_buffer.read().unwrap();

    println!("{:?}", buffer_content.get(0).unwrap());
    println!("baseline {:?}", baseline);

    // let color_calc = ImageAvgColor::new(device, queue, texture, 256);
}
