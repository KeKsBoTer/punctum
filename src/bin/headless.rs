use std::{env, sync::Arc};

use punctum::{render_point_cloud, PointCloud};
fn main() {
    let args = env::args();
    if args.len() != 3 {
        panic!("Usage: <point_cloud>.ply <out_image>.png");
    }
    let arguments = args.collect::<Vec<String>>();
    let ply_file = arguments.get(1).unwrap();
    let img_file = arguments.get(2).unwrap();

    let pc = Arc::new(PointCloud::from_ply_file(ply_file));
    let result = render_point_cloud(pc, 512);

    result.save(img_file).unwrap();
    println!("done!");
}
