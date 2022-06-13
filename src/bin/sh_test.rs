use std::fs;

use image::{ImageBuffer, Rgba};
use punctum::sh::{calc_sh, lm2flat_index};

use punctum::select_physical_device;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, StorageImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::{Sampler, SamplerCreateInfo},
    sync::{self, GpuFuture},
};

fn main2() {
    let l_max = 5;

    let res = 1024;
    let images = calc_sh(l_max, res);
    for (i, img_data) in images.into_iter().enumerate() {
        let data = img_data
            .iter()
            .map(|v| v.to_le_bytes().to_vec())
            .flatten()
            .collect::<Vec<u8>>();

        fs::write(format!("sh_tests/test_{}.bin", i), data).unwrap();
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/bin/sh_test.comp",
    }
}

fn read_coefs() -> Vec<[f32; 4]> {
    let contents = fs::read_to_string("sample_coefs.csv").unwrap();
    let coefs = contents
        .split("\n")
        .filter(|s| s.len() > 0)
        .map(|line| {
            let n = line
                .split(" ")
                .map(|s| s.parse::<f32>().unwrap())
                .collect::<Vec<f32>>();
            [n[0], n[1], n[2], 1.]
        })
        .collect::<Vec<[f32; 4]>>();
    coefs
}

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

    let img_size = 1024;
    let lmax = 10;

    let images = calc_sh(lmax, img_size);
    println!("num shs: {}", images.len());
    let dimensions = ImageDimensions::Dim2d {
        width: img_size,
        height: img_size,
        array_layers: images.len() as u32,
    };

    let render_size = 1024;

    let (sh_images, future) = ImmutableImage::from_iter(
        images.into_iter().flatten().collect::<Vec<f32>>(),
        dimensions,
        vulkano::image::MipmapsCount::One,
        Format::R32_SFLOAT,
        queue.clone(),
    )
    .unwrap();

    sync::now(device.clone())
        .join(future)
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let sh_images_view = ImageView::new_default(sh_images.clone()).unwrap();

    let sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

    let target_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: render_size,
            height: render_size,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        [queue.family()],
    )
    .unwrap();

    let cpu_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::transfer_destination(),
        false,
        (0..render_size * render_size * 4).map(|_| 0u8),
    )
    .unwrap();

    let (local_size_x, local_size_y) = match device.physical_device().properties().subgroup_size {
        Some(subgroup_size) => (32, subgroup_size / 2),
        // Using fallback constant
        None => (8, 8),
    };

    let spec_consts = cs::SpecializationConstants {
        constant_0: local_size_x,
        constant_1: local_size_y,
    };

    let shader = cs::load(device.clone()).unwrap();

    let pipeline = ComputePipeline::new(
        device.clone(),
        shader.entry_point("main").unwrap(),
        &spec_consts,
        None,
        |_| {},
    )
    .unwrap();

    let layout = pipeline.layout();

    let ds_layout = layout.set_layouts().get(0).unwrap();

    let target_img_view = ImageView::new_default(target_image.clone()).unwrap();

    for l in 0..(lmax + 1) {
        let coefs = &read_coefs()[0..lm2flat_index(l, l) + 1];

        let coef_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::storage_buffer(),
            false,
            coefs.iter().map(|v| *v),
        )
        .unwrap();

        let set = PersistentDescriptorSet::new(
            ds_layout.clone(),
            [
                WriteDescriptorSet::image_view_sampler(0, sh_images_view.clone(), sampler.clone()),
                WriteDescriptorSet::image_view(1, target_img_view.clone()),
                WriteDescriptorSet::buffer(2, coef_buffer.clone()),
            ],
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let group_count_x = if render_size / spec_consts.constant_0 > 0 {
            render_size / spec_consts.constant_0
        } else {
            1
        };
        let group_count_y = if render_size / spec_consts.constant_1 > 0 {
            render_size / spec_consts.constant_1
        } else {
            1
        };

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set)
            .dispatch([group_count_x, group_count_y, 1])
            .unwrap()
            .copy_image_to_buffer(target_image.clone(), cpu_buffer.clone())
            .unwrap();

        let command_buffer = builder.build().unwrap();

        sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();

        let buffer_content = cpu_buffer.read().unwrap();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..]).unwrap();
        image.save(format!("sh_tests/image_{}.png", l)).unwrap();
    }
}
