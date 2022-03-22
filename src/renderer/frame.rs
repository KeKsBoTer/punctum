use std::sync::Arc;

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer, SubpassContents,
    },
    device::{physical::PhysicalDevice, Device, DeviceOwned, Queue},
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    pipeline::graphics::viewport::Viewport,
    render_pass::{FramebufferCreateInfo, RenderPass},
    swapchain::{self, AcquireError, Surface, SwapchainAcquireFuture, SwapchainCreateInfo},
    sync::{self, FlushError, GpuFuture},
};
use winit::{dpi::PhysicalSize, window::Window};

use vulkano::render_pass::Framebuffer as VulkanFramebuffer;
use vulkano::swapchain::Swapchain as VulkanSwapchain;

pub struct SurfaceFrame {
    swapchain: Swapchain,
    surface: Arc<Surface<Window>>,
    viewport: Viewport,
    render_pass: Arc<RenderPass>,
    buffers: Vec<Framebuffer<SwapchainImage<Window>>>,

    recreate_swapchain: bool,
}

impl SurfaceFrame {
    pub fn new(
        surface: Arc<Surface<Window>>,
        device: Arc<Device>,
        physical_device: PhysicalDevice,
        render_pass: Arc<RenderPass>,
        swapchain_format: vulkano::format::Format,
    ) -> Self {
        let win_size = surface.window().inner_size();

        let sc = Swapchain::new(surface.clone(), device, physical_device, swapchain_format);
        let fbs = sc.create_framebuffers(&render_pass);

        SurfaceFrame {
            swapchain: sc,
            surface: surface,
            viewport: SurfaceFrame::get_viewport(win_size),
            render_pass: render_pass,
            buffers: fbs,

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
        self.viewport = SurfaceFrame::get_viewport(new_size);
        self.recreate_swapchain = true;
    }

    pub fn viewport(&self) -> &Viewport {
        &self.viewport
    }

    pub fn recreate_if_necessary(&mut self) {
        if self.recreate_swapchain {
            let size = self.surface.window().inner_size();
            self.swapchain.resize(size.into());
            self.buffers = self.swapchain.create_framebuffers(&self.render_pass);
            self.recreate_swapchain = false;
        }
    }

    pub fn render(&mut self, queue: Arc<Queue>, cb: Arc<SecondaryAutoCommandBuffer>) {
        let device = self.swapchain.device();
        let (image_i, suboptimal, acquire_future) = match self.swapchain.acquire_next_image() {
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
                fb.buffer.clone(),
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
            .then_swapchain_present(queue.clone(), self.swapchain.sc.clone(), image_i)
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

struct Framebuffer<I>
where
    I: ImageAccess + 'static,
{
    buffer: Arc<vulkano::render_pass::Framebuffer>,
    image: Arc<I>,
}

impl<I> Framebuffer<I>
where
    I: ImageAccess + 'static,
{
    fn new(image: Arc<I>, render_pass: &Arc<RenderPass>) -> Self {
        let image_view = ImageView::new_default(image.clone()).unwrap();

        let depth_buffer = ImageView::new_default(
            AttachmentImage::transient(
                render_pass.device().clone(),
                image.dimensions().width_height(),
                vulkano::format::Format::D16_UNORM,
            )
            .unwrap(),
        )
        .unwrap();

        let buffer = VulkanFramebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![image_view, depth_buffer.clone()],
                ..Default::default()
            },
        )
        .unwrap();
        Framebuffer { buffer, image }
    }
}

struct Swapchain {
    sc: Arc<VulkanSwapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
}

impl Swapchain {
    fn new(
        surface: Arc<Surface<Window>>,
        device: Arc<Device>,
        physical_device: PhysicalDevice,
        format: vulkano::format::Format,
    ) -> Self {
        let surface_capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let (sc, images) = VulkanSwapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,

                image_format: Some(format),
                image_extent: surface.window().inner_size().into(),

                image_usage: ImageUsage::color_attachment(),

                ..Default::default()
            },
        )
        .unwrap();
        Swapchain { sc, images }
    }

    fn recreate(&mut self) {
        let (new_swapchain, new_images) = self.sc.recreate(self.sc.create_info()).unwrap();
        self.sc = new_swapchain;
        self.images = new_images;
    }

    fn resize(&mut self, size: [u32; 2]) {
        let (new_swapchain, new_images) = self
            .sc
            .recreate(SwapchainCreateInfo {
                image_extent: size,
                ..self.sc.create_info()
            })
            .unwrap();
        self.sc = new_swapchain;
        self.images = new_images;
    }

    fn create_framebuffers(
        &self,
        render_pass: &Arc<RenderPass>,
    ) -> Vec<Framebuffer<SwapchainImage<Window>>> {
        self.images
            .iter()
            .map(|image| Framebuffer::new(image.clone(), render_pass))
            .collect::<Vec<_>>()
    }

    fn acquire_next_image(
        &self,
    ) -> Result<(usize, bool, SwapchainAcquireFuture<Window>), AcquireError> {
        swapchain::acquire_next_image(self.sc.clone(), None)
    }

    fn device(&self) -> &Arc<Device> {
        &self.sc.device()
    }
}
