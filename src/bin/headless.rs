use image::Rgba;
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use std::{env, fs::File, sync::Arc};

use punctum::{Camera, OfflineRenderer, PerceivedColor, PointCloud, RenderSettings};
fn main() {
    let args = env::args();
    if args.len() != 3 {
        panic!("Usage: <point_cloud>.ply <output_folder>");
    }
    let arguments = args.collect::<Vec<String>>();
    let ply_file = arguments.get(1).unwrap();
    let output_file = arguments.get(2).unwrap();

    let mut f = std::fs::File::open("sphere.ply").unwrap();

    // create a parser
    let p = ply_rs::parser::Parser::<PerceivedColor>::new();

    // use the parser: read the entire file
    let in_ply = p.read_ply(&mut f).unwrap();

    let cameras = in_ply
        .payload
        .get("vertex")
        .unwrap()
        .clone()
        .iter()
        .map(|c| Camera::on_unit_sphere(c.pos))
        .collect::<Vec<Camera>>();

    let mut pc = PointCloud::from_ply_file(ply_file);
    pc.scale_to_unit_sphere();

    let pc_arc = Arc::new(pc);

    let mut renderer = OfflineRenderer::new(
        pc_arc,
        256,
        RenderSettings {
            point_size: 20.0,
            ..RenderSettings::default()
        },
    );

    let renders: Vec<Rgba<u8>> = cameras.iter().map(|c| renderer.render(c.clone())).collect();

    println!("done rendering ... saving ....");

    let mut ply = {
        let mut ply = Ply::<PerceivedColor>::new();
        ply.header.encoding = Encoding::Ascii;

        ply.header.elements.add(PerceivedColor::element_def());

        let cam_colors: Vec<PerceivedColor> = renders
            .iter()
            .zip(cameras)
            .map(|(color, cam)| PerceivedColor {
                pos: *cam.position(),
                color: *color,
            })
            .collect();

        ply.payload.insert("vertex".to_string(), cam_colors);
        ply
    };

    let w = Writer::new();
    let mut file = File::create(output_file).unwrap();
    w.write_ply(&mut file, &mut ply).unwrap();

    println!("done!");
}
