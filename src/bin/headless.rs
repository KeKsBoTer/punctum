use image::Rgba;
use nalgebra::{Vector3, Vector4};
use ply_rs::{
    ply::{Addable, Encoding, Ply},
    writer::Writer,
};
use std::{env, fs::File, sync::Arc};

use punctum::{Camera, OfflineRenderer, PointCloud, RenderSettings, Vertex};
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
    let p = ply_rs::parser::Parser::<Vertex>::new();

    // use the parser: read the entire file
    let in_ply = p.read_ply(&mut f).unwrap();

    let cameras = in_ply
        .payload
        .get("vertex")
        .unwrap()
        .clone()
        .iter()
        .map(|c| Camera::on_unit_sphere(c.position.into()))
        .collect::<Vec<Camera>>();

    let mut pc = PointCloud::from_ply_file(ply_file);
    pc.scale_to_unit_sphere();

    let pc_arc = Arc::new(pc);

    let mut renderer = OfflineRenderer::new(
        pc_arc.clone(),
        256,
        RenderSettings {
            point_size: 20.0,
            ..RenderSettings::default()
        },
    );

    let renders: Vec<Rgba<u8>> = cameras.iter().map(|c| renderer.render(c.clone())).collect();

    println!("done rendering ... saving ....");

    let mut ply = {
        let mut ply = Ply::<Vertex>::new();
        ply.header.encoding = Encoding::Ascii;

        ply.header
            .elements
            .add(Vertex::element_def("vertex".to_string()));
        ply.header
            .elements
            .add(Vertex::element_def("camera".to_string()));

        let cam_colors: Vec<Vertex> = renders
            .iter()
            .zip(cameras)
            .map(|(color, cam)| Vertex {
                position: *cam.position(),
                normal: Vector3::zeros(),
                color: Vector4::new(
                    color.0[0] as f32 / 255.,
                    color.0[1] as f32 / 255.,
                    color.0[2] as f32 / 255.,
                    color.0[3] as f32 / 255.,
                ),
            })
            .collect();

        let vertices = pc_arc.points().clone();

        ply.payload.insert("camera".to_string(), cam_colors);
        ply.payload.insert("vertex".to_string(), vertices);
        ply
    };

    let w = Writer::new();
    let mut file = File::create(output_file).unwrap();
    w.write_ply(&mut file, &mut ply).unwrap();

    println!("done!");
}
