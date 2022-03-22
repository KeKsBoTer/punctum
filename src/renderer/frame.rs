use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer, SubpassContents,
    },
    device::{physical::PhysicalDevice, Device, DeviceOwned, Queue},
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        self, AcquireError, Surface, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};
use winit::{dpi::PhysicalSize, window::Window};

pub struct Frame {
    swapchain: Arc<Swapchain<Window>>,
    surface: Arc<Surface<Window>>,
    viewport: Viewport,
    render_pass: Arc<RenderPass>,
    buffers: Vec<Arc<Framebuffer>>,

    recreate_swapchain: bool,
}

impl Frame {
    pub fn new(
        surface: Arc<Surface<Window>>,
        device: Arc<Device>,
        physical_device: PhysicalDevice,
        render_pass: Arc<RenderPass>,
        swapchain_format: vulkano::format::Format,
        pass: Arc<RenderPass>,
    ) -> Self {
        let (swapchain, images) = create_swapchain(
            surface.clone(),
            device.clone(),
            physical_device,
            swapchain_format,
        );

        let win_size = surface.window().inner_size();

        Frame {
            swapchain: swapchain,
            surface: surface,
            viewport: Frame::get_viewport(win_size),
            render_pass: render_pass,
            buffers: get_framebuffers(&images, pass),

            recreate_swapchain: false,
        }
    }

    // build viewport with y flip
    // see: https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
    fn get_viewport(size: PhysicalSize<u32>) -> Viewport {
        let win_size: [f32; 2] = size.into();
        return Viewport {
            origin: [0.0, win_size[1]],
            dimensions: [win_size[0], -win_size[1]],
            depth_range: 0.0..1.0,
        };
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.viewport = Frame::get_viewport(new_size);
        self.recreate_swapchain = true;
    }

    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    pub fn recreate_if_necessary(&mut self) {
        let size = self.surface.window().inner_size();
        // TODO change swapchain size
        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: size.into(),
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            // This error tends to happen when the user is manually resizing the window.
            // Simply restarting the loop is the easiest way to fix this issue.
            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };
        self.swapchain = new_swapchain;
        self.buffers = get_framebuffers(&new_images, self.render_pass.clone());
    }

    pub fn render(&mut self, queue: Arc<Queue>, cb: Arc<SecondaryAutoCommandBuffer>) {
        let device = self.swapchain.device();
        let (image_i, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
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
        let fb = &self.buffers[image_i];

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                fb.clone(),
                SubpassContents::SecondaryCommandBuffers,
                vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
            )
            .unwrap();

        builder.execute_commands(cb).unwrap();
        builder.end_render_pass().unwrap();
        let command_buffer = builder.build().unwrap();

        let execution = sync::now(self.swapchain.device().clone())
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), self.swapchain.clone(), image_i)
            .then_signal_fence_and_flush();

        match execution {
            Ok(future) => {
                future.wait(None).unwrap(); // wait for the GPU to finish
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
            }
        }
    }
}

fn get_framebuffers(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            let depth_buffer = ImageView::new_default(
                AttachmentImage::transient(
                    render_pass.device().clone(),
                    image.dimensions().width_height(),
                    vulkano::format::Format::D16_UNORM,
                )
                .unwrap(),
            )
            .unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn create_swapchain(
    surface: Arc<Surface<Window>>,
    device: Arc<Device>,
    physical_device: PhysicalDevice,
    format: vulkano::format::Format,
) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    let surface_capabilities = physical_device
        .surface_capabilities(&surface, Default::default())
        .unwrap();

    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count,

            image_format: Some(format),
            image_extent: surface.window().inner_size().into(),

            image_usage: ImageUsage::color_attachment(),

            // The alpha mode indicates how the alpha value of the final image will behave. For
            // example, you can choose whether the window will be opaque or transparent.
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .iter()
                .next()
                .unwrap(),

            ..Default::default()
        },
    )
    .unwrap()
}
