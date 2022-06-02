use image::io::Reader as ImageReader;
use punctum::{calc_average_color, select_physical_device, ImageAvgColor};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
    format::Format,
    image::{view::ImageView, ImageDimensions, StorageImage},
    instance::{Instance, InstanceCreateInfo},
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
            // .map(|p| {
            //     [
            //         p.0[0] as f32 / 255.,
            //         p.0[1] as f32 / 255.,
            //         p.0[2] as f32 / 255.,
            //         p.0[3] as f32 / 255.,
            //     ]
            // })
            .map(|p| p.0)
            .collect::<Vec<[u8; 4]>>(),
    )
    .unwrap();

    let src_img = StorageImage::new(
        device.clone(),
        dimensions,
        Format::R8G8B8A8_UNORM,
        Some(queue.family()),
    )
    .unwrap();

    let src_img_view = ImageView::new_default(src_img.clone()).unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .copy_buffer_to_image(buffer.clone(), src_img.clone())
        .unwrap();

    let command_buffer = builder.build().unwrap();

    let avg_color_calc =
        ImageAvgColor::new(device.clone(), queue.clone(), src_img_view.clone(), 256);

    let result = avg_color_calc.calc_average_color(command_buffer);

    let src_buffer_content = buffer.read().unwrap();

    let baseline = calc_average_color(&src_buffer_content);

    println!("{:?}", result);
    println!("baseline {:?}", baseline);
}
