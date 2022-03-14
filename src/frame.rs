use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer, SubpassContents,
    },
    device::{physical::PhysicalDevice, Device, DeviceOwned, Queue},
    format::Format,
    image::{view::ImageView, ImageUsage, SwapchainImage},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, RenderPass},
    swapchain::{self, AcquireError, Surface, Swapchain, SwapchainCreationError},
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
        queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        swapchain_format: Format,
        pass: Arc<RenderPass>,
    ) -> Self {
        let (swapchain, images) = create_swapchain(
            surface.clone(),
            device.clone(),
            physical_device,
            queue.clone(),
            swapchain_format,
        );

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: surface.window().inner_size().into(),
            depth_range: 0.0..1.0,
        };
        Frame {
            swapchain: swapchain,
            surface: surface,
            viewport: viewport,
            render_pass: render_pass,
            buffers: get_framebuffers(&images, pass),

            recreate_swapchain: false,
        }
    }

    pub fn recreate(&mut self, new_size: Option<PhysicalSize<u32>>) {
        let size = new_size.unwrap_or(self.surface.window().inner_size().into());
        let (new_swapchain, new_images) =
            match self.swapchain.recreate().dimensions(size.into()).build() {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };
        self.swapchain = new_swapchain;
        self.buffers = get_framebuffers(&new_images, self.render_pass.clone());

        if new_size.is_some() {
            self.viewport.dimensions = size.into();
        }
    }

    pub fn next_frame(&mut self, queue: Arc<Queue>, cb: Arc<SecondaryAutoCommandBuffer>) {
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
                vec![[0.0, 0.0, 1.0, 1.0].into()],
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
            let view = ImageView::new(image.clone()).unwrap();
            Framebuffer::start(render_pass.clone())
                .add(view)
                .unwrap()
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>()
}

fn create_swapchain(
    surface: Arc<Surface<Window>>,
    device: Arc<Device>,
    physical_device: PhysicalDevice,
    queue: Arc<Queue>,
    format: Format,
) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
    let caps = surface
        .capabilities(physical_device)
        .expect("failed to get surface capabilities");
    let dimensions: [u32; 2] = surface.window().inner_size().into();
    let composite_alpha = caps.supported_composite_alpha.iter().next().unwrap();

    Swapchain::start(device.clone(), surface.clone())
        .num_images(caps.min_image_count + 1) // How many buffers to use in the swapchain
        .format(format)
        .dimensions(dimensions)
        .usage(ImageUsage::color_attachment()) // What the images are going to be used for
        .sharing_mode(&queue) // The queue(s) that the resource will be used
        .composite_alpha(composite_alpha)
        .build()
        .expect("failed to create swapchain")
}
