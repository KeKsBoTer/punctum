use std::sync::Arc;

use vulkano::{
    device::DeviceOwned,
    image::{view::ImageView, AttachmentImage, ImageAccess},
    render_pass::{Framebuffer as VulkanFramebuffer, FramebufferCreateInfo, RenderPass},
};
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
    pub fn new(image: Arc<I>, render_pass: Arc<RenderPass>) -> Self {
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
                attachments: vec![image_view.clone(), depth_buffer.clone()],
                ..Default::default()
            },
        )
        .unwrap();
        Framebuffer { buffer, image }
    }

    pub fn vulkan_fb(&self) -> &Arc<vulkano::render_pass::Framebuffer> {
        &self.buffer
    }

    pub fn image(&self) -> &Arc<I> {
        &self.image
    }
}

unsafe impl<I> DeviceOwned for Framebuffer<I>
where
    I: ImageAccess + 'static,
{
    fn device(&self) -> &Arc<vulkano::device::Device> {
        self.buffer.device()
    }
}
