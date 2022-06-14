use std::sync::Arc;

use vulkano::{
    buffer::{cpu_pool::CpuBufferPoolSubbuffer, BufferContents, BufferUsage, CpuBufferPool},
    device::Device,
    memory::pool::StdMemoryPool,
};

pub struct UniformBuffer<T>
where
    [T]: BufferContents,
    T: BufferContents + Default + Copy,
{
    data: T,
    buffer_pool: CpuBufferPool<T, Arc<StdMemoryPool>>,
    pool_chunk: Arc<CpuBufferPoolSubbuffer<T, Arc<StdMemoryPool>>>,
    // buffer: Arc<DeviceLocalBuffer<T>>,
}

impl<T> UniformBuffer<T>
where
    [T]: BufferContents,
    T: BufferContents + Default + Copy,
{
    pub fn new(device: Arc<Device>, data: Option<T>) -> Self {
        let pool = CpuBufferPool::new(device.clone(), BufferUsage::all());
        let pool_chunk = pool.next(data.unwrap_or_default()).unwrap();

        // let uniform_buffer: Arc<DeviceLocalBuffer<T>> = DeviceLocalBuffer::new(
        //     device.clone(),
        //     BufferUsage::uniform_buffer_transfer_destination(),
        //     None,
        // )
        // .unwrap();
        UniformBuffer {
            data: data.unwrap_or_default(),
            buffer_pool: pool,
            pool_chunk: pool_chunk,
            // buffer: uniform_buffer,
        }
    }

    pub fn update(&mut self, data: T) {
        self.data = data;
        self.pool_chunk = self.buffer_pool.next(self.data).unwrap();
    }

    // pub fn buffer(&self) -> &Arc<DeviceLocalBuffer<T>> {
    //     &self.buffer
    // }
    pub fn pool_chunk(&self) -> &Arc<CpuBufferPoolSubbuffer<T, Arc<StdMemoryPool>>> {
        &self.pool_chunk
    }

    pub fn data(&self) -> &T {
        &self.data
    }
}
