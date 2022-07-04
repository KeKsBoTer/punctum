use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::BufReader;
use std::io::Read;
use std::mem;

use nalgebra::Vector4;
use pbr::ProgressBar;
use punctum::Octree;
use punctum::PointCloud;
use punctum::SHVertex;
use punctum::TeeReader;

fn main() -> io::Result<()> {
    let filename = "datasets/neuschwanstein/neuschwanstein_16_8192max.bin";
    let mut octree: Octree<f32, f32> = {
        let in_file = File::open(filename)?;

        let mut pb = ProgressBar::new(in_file.metadata().unwrap().len());

        let mut buf = BufReader::new(in_file);

        pb.message(&format!("decoding {}: ", filename));

        pb.set_units(pbr::Units::Bytes);
        let mut tee = TeeReader::new(&mut buf, &mut pb);

        let octree: Octree<f64, u8> = bincode::deserialize_from(&mut tee).unwrap();

        pb.finish_println("done!");

        octree.into()
    };

    let f = File::open("coefs.raw")?;
    let mut reader = BufReader::new(f);
    let mut buffer = Vec::new();

    reader.read_to_end(&mut buffer)?;

    let id_size = mem::size_of::<u64>();
    let f_size = mem::size_of::<f32>();
    let color_channels = 4;
    let l = 10;
    let coefs: HashMap<u64, Vec<Vector4<f32>>> = buffer
        .chunks_exact(id_size + f_size * (l + 1) * (l + 1) * color_channels)
        .map(|bytes| {
            let mut a: [u8; 8] = Default::default();
            a.copy_from_slice(&bytes[0..8]);
            let id = u64::from_le_bytes(a);

            let mut af: [u8; 4] = Default::default();
            let coefs = bytes[8..]
                .chunks_exact(f_size * 4)
                .map(|bytes| {
                    af.copy_from_slice(&bytes[0..4]);
                    let r = f32::from_le_bytes(af);
                    af.copy_from_slice(&bytes[4..8]);
                    let g = f32::from_le_bytes(af);
                    af.copy_from_slice(&bytes[8..12]);
                    let b = f32::from_le_bytes(af);
                    af.copy_from_slice(&bytes[12..16]);
                    let a = f32::from_le_bytes(af);
                    Vector4::new(r, g, b, a)
                })
                .collect::<Vec<Vector4<f32>>>();
            return (id, coefs);
        })
        .collect();

    for octant in octree.borrow_mut().into_iter() {
        let pc: &PointCloud<f32, f32> = octant.points().into();
        match coefs.get(&octant.id()) {
            Some(g_coefs) => {
                let mut sh_coefs: [Vector4<f32>; 121] = [Vector4::zeros(); 121];
                for i in 0..sh_coefs.len() {
                    sh_coefs[i] = g_coefs[i];
                }
                octant.sh_approximation = Some(SHVertex::new(pc.mean().position, sh_coefs.into()));
            }
            None => panic!("not found: {}", octant.id()),
        }
    }
    Ok(())
}
