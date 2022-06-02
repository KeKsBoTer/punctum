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
    shader::{SpecializationConstants, SpecializationMapEntry},
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
}

const IMG_SIZES: [u32; 11] = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1];

#[repr(C)]
struct ShaderGroupSize {
    x: u32,
    y: u32,
}

unsafe impl SpecializationConstants for ShaderGroupSize {
    #[inline]
    fn descriptors() -> &'static [SpecializationMapEntry] {
        static DESCRIPTORS: [SpecializationMapEntry; 2] = [
            SpecializationMapEntry {
                constant_id: 0,
                offset: 0,
                size: 4,
            },
            SpecializationMapEntry {
                constant_id: 1,
                offset: 4,
                size: 4,
            },
        ];

        &DESCRIPTORS
    }
}

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

        let spec_consts = ShaderGroupSize { x: 32, y: 32 };

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

            let group_count_x = if sizes[i] / spec_consts.x > 0 {
                sizes[i] / spec_consts.x
            } else {
                1
            };
            let group_count_y = if sizes[i] / spec_consts.y > 0 {
                sizes[i] / spec_consts.y
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
        }
    }

    pub fn calc_average_color(&self, cb_before: PrimaryAutoCommandBuffer) -> Rgba<u8> {
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), cb_before)
            .unwrap()
            .then_execute(self.queue.clone(), self.command_buffer.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();

        let buffer_content = self.target_buffer.read().unwrap();

        let rgba = buffer_content[0];
        return Rgba([
            (rgba[0] * 255.) as u8,
            (rgba[1] * 255.) as u8,
            (rgba[2] * 255.) as u8,
            (rgba[3] * 255.) as u8,
        ]);
    }
}
