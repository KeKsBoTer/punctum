use std::sync::Arc;

use vulkano::{
    device::{physical::PhysicalDevice, Device, DeviceOwned},
    image::{ImageUsage, SwapchainImage},
    render_pass::RenderPass,
    swapchain::{
        self, AcquireError, Surface, Swapchain as VulkanSwapchain, SwapchainAcquireFuture,
        SwapchainCreateInfo,
    },
};
use winit::window::Window;

use super::Framebuffer;

pub struct Swapchain {
    sc: Arc<VulkanSwapchain<Window>>,
    buffers: Vec<Framebuffer<SwapchainImage<Window>>>,
    render_pass: Arc<RenderPass>,
}

impl Swapchain {
    pub fn new(
        surface: Arc<Surface<Window>>,
        device: Arc<Device>,
        physical_device: PhysicalDevice,
        format: vulkano::format::Format,
        render_pass: Arc<RenderPass>,
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
                // present_mode: PresentMode::Immediate,
                ..Default::default()
            },
        )
        .unwrap();
        Swapchain {
            sc,
            buffers: Swapchain::create_framebuffers(&images, render_pass.clone()),
            render_pass: render_pass,
        }
    }

    pub fn resize(&mut self, size: [u32; 2]) {
        let (new_swapchain, new_images) = self
            .sc
            .recreate(SwapchainCreateInfo {
                image_extent: size,
                ..self.sc.create_info()
            })
            .unwrap();
        self.sc = new_swapchain;
        self.buffers = Swapchain::create_framebuffers(&new_images, self.render_pass.clone());
    }

    fn create_framebuffers(
        images: &Vec<Arc<SwapchainImage<Window>>>,
        render_pass: Arc<RenderPass>,
    ) -> Vec<Framebuffer<SwapchainImage<Window>>> {
        images
            .iter()
            .map(|image| Framebuffer::new(image.clone(), render_pass.clone()))
            .collect::<Vec<_>>()
    }

    pub fn acquire_next_image(
        &self,
    ) -> Result<
        (
            &Framebuffer<SwapchainImage<Window>>,
            usize,
            bool,
            SwapchainAcquireFuture<Window>,
        ),
        AcquireError,
    > {
        match swapchain::acquire_next_image(self.sc.clone(), None) {
            Ok((i, suboptimal, future)) => Ok((&self.buffers[i], i, suboptimal, future)),
            Err(e) => Err(e),
        }
    }

    pub fn vk_swapchain(&self) -> &Arc<VulkanSwapchain<Window>> {
        &self.sc
    }
}

unsafe impl DeviceOwned for Swapchain {
    fn device(&self) -> &Arc<vulkano::device::Device> {
        self.sc.device()
    }
}
