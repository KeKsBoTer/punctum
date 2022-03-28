use std::{env, sync::Arc};

use punctum::{render_point_cloud, PointCloud, RenderSettings};
fn main() {
    let args = env::args();
    if args.len() != 3 {
        panic!("Usage: <point_cloud>.ply <out_image>.png");
    }
    let arguments = args.collect::<Vec<String>>();
    let ply_file = arguments.get(1).unwrap();
    let img_file = arguments.get(2).unwrap();

    let cameras = punctum::Camera::load_from_ply("sphere.ply");

    let mut pc = PointCloud::from_ply_file(ply_file);
    pc.scale_to_unit_sphere();
    println!("pc box: {:?}", pc.bounding_box());
    let pc_arc = Arc::new(pc);
    for (i, c) in cameras.iter().enumerate() {
        // println!("camera: {:?}", c);
        let result = render_point_cloud(
            pc_arc.clone(),
            256,
            c.clone(),
            RenderSettings {
                point_size: 5.0,
                ..RenderSettings::default()
            },
        );
        result.save(format!("renders/render_{:}.png", i)).unwrap();
    }
    println!("done!");
}
