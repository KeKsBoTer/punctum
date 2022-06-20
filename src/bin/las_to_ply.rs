use las::{Read as LasRead, Reader};
use nalgebra::{center, Point3, Vector4};
use pbr::ProgressBar;
use punctum::{export_ply, PointCloud, Vertex};
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "Laz to Ply test")]
struct Opt {
    #[structopt(name = "input_las", parse(from_os_str))]
    input: PathBuf,

    #[structopt(name = "output", parse(from_os_str))]
    output: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    let mut reader = Reader::from_path(opt.input).unwrap();

    let number_of_points = reader.header().number_of_points();

    let bounds = reader.header().bounds();
    let min_point = Point3::new(bounds.min.x, bounds.min.y, bounds.min.z);
    let max_point = Point3::new(bounds.max.x, bounds.max.y, bounds.max.z);
    let size = max_point - min_point;
    let max_size = size[size.imax()];

    let center = center(&min_point, &max_point);

    let mut pb = ProgressBar::new(number_of_points);
    let mut counter = 0;

    let mut points = Vec::new();

    for p in reader.points() {
        counter += 1;

        let point = p.unwrap();
        let color = point.color.unwrap();
        let point = Vertex {
            position: Point3::new(point.x, point.y, point.z),
            color: Vector4::new(
                (color.red / 256) as u8, // 65536 = 2**16
                (color.green / 256) as u8,
                (color.blue / 256) as u8,
                255,
            ),
        };

        if (point.position - center).norm_squared() < (max_size * max_size / (10. * 10.)) {
            points.push(point);
        }

        if counter >= number_of_points / 100 {
            pb.add(counter);
            counter = 0;
        }
    }
    pb.add(counter);

    let pc = PointCloud::from_vec(&points);

    export_ply(&opt.output, &pc);
}
