use std::sync::Arc;

use image::Rgba;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImageViewAbstract, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::Filter,
    sync::{self, GpuFuture},
};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/renderer/shaders/image_reduction.comp"
    }
}

pub struct ImageAvgColor {
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
    target_buffer: Arc<CpuAccessibleBuffer<[[f32; 4]]>>,
    img_size: u32,
}

const IMG_SIZES: [u32; 11] = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1];

impl ImageAvgColor {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        src_img: Arc<dyn ImageViewAbstract>,
        start_size: u32,
    ) -> Self {
        assert!(IMG_SIZES.contains(&start_size));

        let sizes: Vec<u32> = IMG_SIZES.into_iter().filter(|s| *s <= start_size).collect();

        let images: Vec<Arc<ImageView<StorageImage>>> = sizes
            .iter()
            .map(|s| {
                let img = StorageImage::new(
                    device.clone(),
                    ImageDimensions::Dim2d {
                        width: *s,
                        height: *s,
                        array_layers: 1,
                    },
                    Format::R32G32B32A32_SFLOAT,
                    Some(queue.family()),
                )
                .unwrap();

                ImageView::new_default(img).unwrap()
            })
            .collect();

        let target_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::transfer_destination(),
            false,
            (0..1).map(|_| [0f32; 4]),
        )
        .unwrap();

        let shader = cs::load(device.clone()).unwrap();

        let (local_size_x, local_size_y) = match device.physical_device().properties().subgroup_size
        {
            Some(subgroup_size) => (32, subgroup_size / 2),
            // Using fallback constant
            None => (8, 8),
        };

        let spec_consts = cs::SpecializationConstants {
            constant_0: local_size_x,
            constant_1: local_size_y,
        };

        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &spec_consts,
            None,
            |_| {},
        )
        .unwrap();

        let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        let size = start_size as i32;
        // convert image type from rgba8 to rgba32f by using blit command
        builder
            .blit_image(
                src_img.image(),
                [0, 0, 0],
                [size, size, 1],
                0,
                0,
                images[0].image().clone(),
                [0, 0, 0],
                [size, size, 1],
                0,
                0,
                1,
                Filter::Nearest,
            )
            .unwrap();
        builder.bind_pipeline_compute(compute_pipeline.clone());

        for i in 0..sizes.len() - 1 {
            let set = PersistentDescriptorSet::new(
                layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, images[i].clone()),
                    WriteDescriptorSet::image_view(1, images[i + 1].clone()),
                ],
            )
            .unwrap();

            let layout = compute_pipeline.layout();
            let push_constants = cs::ty::PushConstantData {
                first: (i == 0) as u32,
            };

            let group_count_x = if sizes[i] / spec_consts.constant_0 > 0 {
                sizes[i] / spec_consts.constant_0
            } else {
                1
            };
            let group_count_y = if sizes[i] / spec_consts.constant_1 > 0 {
                sizes[i] / spec_consts.constant_1
            } else {
                1
            };

            builder
                .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set)
                .push_constants(layout.clone(), 0, push_constants)
                .dispatch([group_count_x, group_count_y, 1])
                .unwrap();
        }

        builder
            .copy_image_to_buffer(
                images.last().unwrap().image().clone(),
                target_buffer.clone(),
            )
            .unwrap();

        let command_buffer = Arc::new(builder.build().unwrap());

        ImageAvgColor {
            device,
            queue,
            command_buffer,
            target_buffer,
            img_size: start_size,
        }
    }

    pub fn calc_average_color(&self, cb_before: PrimaryAutoCommandBuffer) -> Rgba<u8> {
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cb_before)
            .unwrap()
            .then_execute_same_queue(self.command_buffer.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        let buffer_content = self.target_buffer.read().unwrap();

        let rgba = buffer_content[0];

        let red = rgba[0] / rgba[3];
        let green = rgba[1] / rgba[3];
        let blue = rgba[2] / rgba[3];
        let alpha = rgba[3] / (self.img_size * self.img_size) as f32;

        return Rgba([
            (red * 255.) as u8,
            (green * 255.) as u8,
            (blue * 255.) as u8,
            (alpha * 255.) as u8,
        ]);
    }
}

#[cfg(test)]
mod tests {
    use image::{io::Reader as ImageReader, Rgba};
    use nalgebra::{vector, Vector4};
    use rayon::{iter::ParallelIterator, slice::ParallelSlice};
    use vulkano::{
        buffer::{BufferUsage, CpuAccessibleBuffer},
        command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
        device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo},
        format::Format,
        image::{view::ImageView, ImageDimensions, StorageImage},
        instance::{Instance, InstanceCreateInfo},
    };

    use crate::{select_physical_device, ImageAvgColor};

    #[test]
    fn cpu_eq_gpu() {
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

        let baseline = calc_average_color_cpu(&src_buffer_content);

        // allow for a difference of one to account for rounding errors
        let diff =
            (Vector4::from(result.0).cast::<i32>() - Vector4::from(baseline.0).cast::<i32>()).abs();
        assert!(
            diff.amax() <= 1,
            "cpu ({:?}) != gpu ({:?})",
            baseline,
            result,
        );
    }

    pub fn calc_average_color_cpu(data: &[[u8; 4]]) -> Rgba<u8> {
        let start = vector!(0., 0., 0., 0.);
        let rgba_sum = data
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

        let red = rgba_sum.x / rgba_sum.w;
        let green = rgba_sum.y / rgba_sum.w;
        let blue = rgba_sum.z / rgba_sum.w;
        let alpha = rgba_sum.w / (data.len()) as f32;

        return Rgba([
            (red * 255.) as u8,
            (green * 255.) as u8,
            (blue * 255.) as u8,
            (alpha * 255.) as u8,
        ]);
    }
}
