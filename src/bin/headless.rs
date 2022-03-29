use image::{ImageBuffer, Rgba};
use rayon::prelude::*;
use std::{env, sync::Arc};

use punctum::{OfflineRenderer, PointCloud, RenderSettings};
fn main() {
    let args = env::args();
    if args.len() != 3 {
        panic!("Usage: <point_cloud>.ply <output_folder>");
    }
    let arguments = args.collect::<Vec<String>>();
    let ply_file = arguments.get(1).unwrap();
    let output_folder = arguments.get(2).unwrap();

    let cameras = punctum::Camera::load_from_ply("sphere.ply");

    let mut pc = PointCloud::from_ply_file(ply_file);
    pc.scale_to_unit_sphere();
    println!("pc box: {:?}", pc.bounding_box());

    let pc_arc = Arc::new(pc);

    let mut renderer = OfflineRenderer::new(
        pc_arc,
        256,
        RenderSettings {
            point_size: 5.0,
            ..RenderSettings::default()
        },
    );

    let renders: Vec<ImageBuffer<Rgba<u8>, Vec<u8>>> =
        cameras.iter().map(|c| renderer.render(c.clone())).collect();

    println!("done rendering ... saving ....");
    renders.par_iter().enumerate().for_each(|(i, img)| {
        img.save(format!("{:}/render_{:}.png", output_folder, i))
            .unwrap()
    });
    println!("done!");
}
