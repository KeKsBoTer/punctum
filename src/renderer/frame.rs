use std::sync::Arc;

use vulkano::{
    buffer::CpuAccessibleBuffer,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, SecondaryAutoCommandBuffer, SubpassContents,
    },
    device::{physical::PhysicalDevice, Device, DeviceOwned, Queue},
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageDimensions, ImageUsage, StorageImage,
        SwapchainImage,
    },
    render_pass::{FramebufferCreateInfo, RenderPass},
    swapchain::{self, AcquireError, Surface, SwapchainAcquireFuture, SwapchainCreateInfo},
    sync::{self, FlushError, GpuFuture},
};
use winit::window::Window;

use vulkano::pipeline::graphics::viewport::Viewport as VulkanViewport;
use vulkano::render_pass::Framebuffer as VulkanFramebuffer;
use vulkano::swapchain::Swapchain as VulkanSwapchain;

pub struct Frame {
    pub buffer: Framebuffer<StorageImage>,
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
        let fb = Framebuffer::new(image, &render_pass);
        Frame { buffer: fb }
    }

    pub fn render(
        &mut self,
        queue: Arc<Queue>,
        cb: Arc<SecondaryAutoCommandBuffer>,
        target_buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    ) {
        let fb = &self.buffer;

        let device = self.buffer.buffer.device();

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
        builder
            .copy_image_to_buffer(fb.image.clone(), target_buffer.clone())
            .unwrap();
        let command_buffer = builder.build().unwrap();

        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();
    }
}

pub struct SurfaceFrame {
    buffers: Vec<Framebuffer<SwapchainImage<Window>>>,

    swapchain: Swapchain,
    surface: Arc<Surface<Window>>,
    render_pass: Arc<RenderPass>,

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
        let sc = Swapchain::new(surface.clone(), device, physical_device, swapchain_format);
        let fbs = sc.create_framebuffers(&render_pass);

        SurfaceFrame {
            swapchain: sc,
            surface: surface,
            render_pass: render_pass,
            buffers: fbs,

            recreate_swapchain: false,
        }
    }

    pub fn force_recreate(&mut self) {
        self.recreate_swapchain = true;
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
        // PREPARE BUFFER
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

        // record command buffer

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

        // submit command buffer

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

pub struct Framebuffer<I>
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
#[derive(Debug, Clone)]
pub struct Viewport {
    vp: VulkanViewport,
}

impl Viewport {
    pub fn new(size: [u32; 2]) -> Self {
        Viewport {
            vp: Viewport::build_vulkan_viewport(size),
        }
    }

    // build viewport with y flip
    // see: https://www.saschawillems.de/blog/2019/03/29/flipping-the-vulkan-viewport/
    fn build_vulkan_viewport(size: [u32; 2]) -> VulkanViewport {
        VulkanViewport {
            origin: [0.0, size[1] as f32],
            dimensions: [size[0] as f32, -(size[1] as f32)],
            depth_range: 0.0..1.0,
        }
    }

    pub fn resize(&mut self, size: [u32; 2]) {
        self.vp = Viewport::build_vulkan_viewport(size);
    }

    pub fn size(&self) -> [f32; 2] {
        [self.vp.dimensions[0], self.vp.dimensions[1].abs()]
    }
}
impl From<Viewport> for VulkanViewport {
    fn from(viewport: Viewport) -> Self {
        viewport.vp
    }
}
