use std::sync::Arc;

use egui_winit_vulkano::Gui;
use vulkano::{
    buffer::CpuAccessibleBuffer,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        SecondaryAutoCommandBuffer, SubpassContents,
    },
    device::{physical::PhysicalDevice, Device, DeviceOwned, Queue},
    image::{view::ImageView, ImageDimensions, StorageImage, SwapchainImage},
    render_pass::RenderPass,
    swapchain::{AcquireError, Surface},
    sync::{self, FlushError, GpuFuture},
};
use winit::window::Window;

use super::{Framebuffer, Swapchain};

pub struct Frame {
    pub buffer: Framebuffer<StorageImage>,

    background_color: [f32; 4],
}

impl Frame {
    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        image_format: vulkano::format::Format,
        image_size: [u32; 2],
    ) -> Self {
        let image = StorageImage::new(
            device,
            ImageDimensions::Dim2d {
                width: image_size[0],
                height: image_size[1],
                array_layers: 1,
            },
            image_format,
            None,
        )
        .unwrap();
        let fb = Framebuffer::new(image, render_pass);
        Frame {
            buffer: fb,
            background_color: [0.; 4],
        }
    }

    pub fn set_background(&mut self, color: [f32; 4]) {
        self.background_color = color;
    }

    pub fn render(
        &mut self,
        queue: Arc<Queue>,
        cb: Arc<SecondaryAutoCommandBuffer>,
        target_buffer: Option<Arc<CpuAccessibleBuffer<[u8]>>>,
    ) -> PrimaryAutoCommandBuffer {
        let fb = &self.buffer;

        let device = self.buffer.device();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                fb.vulkan_fb().clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![self.background_color.into(), 1f32.into()],
            )
            .unwrap();

        builder.execute_commands(cb).unwrap();
        builder.end_render_pass().unwrap();
        if let Some(buffer) = target_buffer {
            builder
                .copy_image_to_buffer(fb.image().clone(), buffer.clone())
                .unwrap();
        }
        return builder.build().unwrap();
    }
}
pub struct SurfaceFrame {
    swapchain: Swapchain,
    surface: Arc<Surface<Window>>,

    recreate_swapchain: bool,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl SurfaceFrame {
    pub fn new(
        surface: Arc<Surface<Window>>,
        device: Arc<Device>,
        physical_device: PhysicalDevice,
        render_pass: Arc<RenderPass>,
        swapchain_format: vulkano::format::Format,
    ) -> Self {
        let sc = Swapchain::new(
            surface.clone(),
            device.clone(),
            physical_device,
            swapchain_format,
            render_pass,
        );

        let previous_frame_end = Some(sync::now(device.clone()).boxed());

        SurfaceFrame {
            swapchain: sc,
            surface: surface,

            recreate_swapchain: false,
            previous_frame_end,
        }
    }

    pub fn force_recreate(&mut self) {
        self.recreate_swapchain = true;
    }

    pub fn recreate_if_necessary(&mut self) {
        if self.recreate_swapchain {
            let size = self.surface.window().inner_size();
            self.swapchain.resize(size.into());
            self.recreate_swapchain = false;
        }
    }

    pub fn render(
        &mut self,
        queue: Arc<Queue>,
        cb: Arc<SecondaryAutoCommandBuffer>,
        gui: &mut Gui,
    ) {
        let device = self.swapchain.device();
        let (fb, image_i, suboptimal, acquire_future) = match self.swapchain.acquire_next_image() {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swapchain = true;
                return;
            }
            Err(e) => panic!("Failed to acquire next image: {:?}", e),
        };
        if suboptimal {
            self.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                fb.vulkan_fb().clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
            )
            .unwrap();

        builder.execute_commands(cb).unwrap();
        builder.end_render_pass().unwrap();
        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap();

        let after_future = gui.draw_on_image(future, fb.image_view().clone());

        let present_future = after_future
            .then_swapchain_present(
                queue.clone(),
                self.swapchain.vk_swapchain().clone(),
                image_i,
            )
            .then_signal_fence_and_flush();

        match present_future {
            Ok(future) => {
                match future.wait(None) {
                    Ok(x) => x,
                    Err(err) => println!("err: {:?}", err),
                }
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        };
    }

    pub fn render_fn<F>(&mut self, queue: Arc<Queue>, f: F)
    where
        F: Fn(Box<dyn GpuFuture>, Arc<ImageView<SwapchainImage<Window>>>) -> Box<dyn GpuFuture>,
    {
        let device = self.swapchain.device();
        let (fb, image_i, suboptimal, acquire_future) = match self.swapchain.acquire_next_image() {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swapchain = true;
                return;
            }
            Err(e) => panic!("Failed to acquire next image: {:?}", e),
        };
        if suboptimal {
            self.recreate_swapchain = true;
        }

        let swapchain_future = self.previous_frame_end.take().unwrap().join(acquire_future);

        let render_future = f(swapchain_future.boxed(), fb.image_view().clone());

        let present_future = render_future
            .then_swapchain_present(
                queue.clone(),
                self.swapchain.vk_swapchain().clone(),
                image_i,
            )
            .then_signal_fence_and_flush();

        match present_future {
            Ok(future) => {
                match future.wait(None) {
                    Ok(x) => x,
                    Err(err) => println!("err: {:?}", err),
                }
                self.previous_frame_end = Some(future.boxed());
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                self.previous_frame_end = Some(sync::now(device.clone()).boxed());
            }
        };
    }
}
