use std::io::{Read, Stdout, Write};

use pbr::ProgressBar;

/// Tee (T-Split) Writer writes the same data to two child writers.
pub struct TeeWriter<'a, T1: Write + 'a> {
    w1: &'a mut T1,
    w2: &'a mut ProgressBar<Stdout>,
    counter: u64,
}

impl<'a, T1> TeeWriter<'a, T1>
where
    T1: Write + 'a,
{
    pub fn new(w1: &'a mut T1, w2: &'a mut ProgressBar<Stdout>) -> Self {
        Self { w1, w2, counter: 0 }
    }
}

impl<'a, T1> Write for TeeWriter<'a, T1>
where
    T1: Write + 'a,
{
    #[inline]
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let size1 = self.w1.write(buf)?;
        self.counter += size1 as u64;
        // update only every 10Mb
        if self.counter >= 1024 * 1024 * 10 {
            self.w2.add(self.counter);
            self.counter = 0;
        }
        Ok(size1)
    }

    #[inline]
    fn flush(&mut self) -> std::io::Result<()> {
        self.w2.add(self.counter);
        self.counter = 0;
        self.w1.flush()
    }
}

/// Tee (T-Split) Writer writes the same data to two child writers.
pub struct TeeReader<'a, T1: Read + 'a> {
    w1: &'a mut T1,
    w2: &'a mut ProgressBar<Stdout>,
    counter: usize,
}

impl<'a, T1> TeeReader<'a, T1>
where
    T1: Read + 'a,
{
    pub fn new(w1: &'a mut T1, w2: &'a mut ProgressBar<Stdout>) -> Self {
        Self { w1, w2, counter: 0 }
    }
}

impl<'a, T1> Read for TeeReader<'a, T1>
where
    T1: Read + 'a,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.w1.read(buf)?;
        self.counter += n;
        if self.counter >= 1024 * 1024 * 10 {
            self.w2.add(self.counter as u64);
            self.counter = 0;
        }
        Ok(n)
    }
}
