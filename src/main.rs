extern crate image;
extern crate rayon;

mod common_geometry;
mod delaunay;
mod float_ord;
mod halton;
mod rasterization;
mod tree_2d;

use crate::common_geometry::Triangle2D;
use crate::float_ord::FloatOrd;
use rayon::prelude::*;
use std::cmp;
use std::env::args_os;
use std::ffi::OsString;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::halton::HaltonSequence;
use crate::tree_2d::{Coord, Point2D};

fn convert_to_luma(color: &image::Rgb<u8>) -> Coord {
    0.2126 * color.data[0] as Coord
        + 0.7152 * color.data[1] as Coord
        + 0.0722 * color.data[2] as Coord
}

fn generate_initial_points(
    diff_img: &Vec<Coord>,
    n: usize,
    width: u32,
    height: u32,
) -> Vec<Point2D> {
    let f_width = width as Coord;
    let f_height = height as Coord;
    let mut ret = Vec::with_capacity(n);
    let mut halton_x = HaltonSequence::new(2);
    let mut halton_y = HaltonSequence::new(3);
    let mut halton_z = HaltonSequence::new(5);
    let mut num_pushed = 0;
    let scale = diff_img.iter().map(FloatOrd).max().unwrap().0;
    while num_pushed < n {
        let x = halton_x.next().unwrap() * f_width;
        let y = halton_y.next().unwrap() * f_height;
        let z = halton_z.next().unwrap();
        if diff_img[y as usize * width as usize + x as usize] > (z * z) * scale {
            ret.push((x, y));
            num_pushed += 1;
        }
    }
    ret
}

fn compute_differential_image(img: &image::RgbImage) -> Vec<Coord> {
    let mut ret = vec![0.0; (img.width() * img.height()) as usize];
    ret.par_iter_mut().enumerate().for_each(|(id, cell)| {
        let x = id as u32 % img.width();
        let y = id as u32 / img.width();
        let mut diff_x = 0.0;
        let mut diff_y = 0.0;
        let luma = convert_to_luma(img.get_pixel(x, y));
        if x < img.width() - 1 {
            diff_x += (convert_to_luma(img.get_pixel(x + 1, y)) - luma).abs();
        }
        if y < img.height() - 1 {
            diff_y += (convert_to_luma(img.get_pixel(x, y + 1)) - luma).abs();
        }
        if x > 0 {
            diff_x += (luma - convert_to_luma(img.get_pixel(x - 1, y))).abs();
        }
        if y > 0 {
            diff_y += (luma - convert_to_luma(img.get_pixel(x, y - 1))).abs();
        }
        *cell = diff_x * diff_x + diff_y * diff_y;
    });

    let im_width = img.width();
    let im_height = img.height();

    // Blur the image
    const BLUR_RADIUS: i32 = 5;
    // let mut ret2 = Vec::with_capacity((im_width * im_height) as usize);
    let mut ret2 = vec![0.0; (im_width * im_height) as usize];
    ret2.par_iter_mut().enumerate().for_each(|(id, cell)| {
        let x = id as i32 % im_width as i32;
        let y = id as i32 / im_width as i32;
        let mut acc = 0.0;
        for by in -BLUR_RADIUS..BLUR_RADIUS + 1 {
            let oy = cmp::max(0, cmp::min(im_height as i32 - 1, y + by));
            let idx = oy as u32 * im_width + x as u32;
            acc += ret[idx as usize];
        }
        *cell = acc;
    });

    let mut ret3 = vec![0.0; (im_width * im_height) as usize];
    ret3.par_iter_mut().enumerate().for_each(|(id, cell)| {
        let x = id as i32 % im_width as i32;
        let y = id as i32 / im_width as i32;
        let mut acc = 0.0;
        for bx in -BLUR_RADIUS..BLUR_RADIUS + 1 {
            let ox = cmp::max(0, cmp::min(im_width as i32 - 1, x + bx));
            let idx = y as u32 * im_width + ox as u32;
            acc += ret2[idx as usize];
        }
        *cell = acc;
    });

    ret3
}

fn generate_delaunay_image(img: &mut image::RgbImage, triangulation: &Vec<Triangle2D>) {
    let im_width = img.width() as usize;
    let im_height = img.height() as usize;
    let events = rasterization::prepare_events(triangulation);
    let mut tri_stats = Vec::with_capacity(triangulation.len());

    tri_stats.resize(triangulation.len(), ((0.0, 0.0, 0.0), 0));
    let gather_pixel = |x: u32, y: u32, tri_id: Option<usize>| match tri_id {
        Some(id) => {
            let pix = img.get_pixel(x, y);
            let tstats = &mut tri_stats[id];
            (tstats.0).0 += pix.data[0] as Coord;
            (tstats.0).1 += pix.data[1] as Coord;
            (tstats.0).2 += pix.data[2] as Coord;
            tstats.1 += 1;
        }
        None => {}
    };

    rasterization::rasterize(&events, im_width, im_height, gather_pixel);

    for tstat in tri_stats.iter_mut() {
        (tstat.0).0 /= tstat.1 as Coord;
        (tstat.0).1 /= tstat.1 as Coord;
        (tstat.0).2 /= tstat.1 as Coord;
    }

    let mut pixel_id = 0;
    let set_pixel = |x: u32, y: u32, tri_id: Option<usize>| {
        let pixel = match tri_id {
            Some(id) => {
                let tstat = tri_stats[id];
                image::Rgb {
                    data: [(tstat.0).0 as u8, (tstat.0).1 as u8, (tstat.0).2 as u8],
                }
            }
            None => image::Rgb { data: [0; 3] },
        };
        img.put_pixel(x, y, pixel);
        pixel_id += 1;
    };

    rasterization::rasterize(&events, im_width, im_height, set_pixel);
}

// fn draw_distribution(points: &Vec<Point2D>) -> image::RgbImage {
//     let mut img =
// }

fn duration_as_seconds(d: Duration) -> f64 {
    d.as_secs() as f64 + d.subsec_micros() as f64 / 1e6
}

struct Timer {
    last_timestamp: Instant,
}

impl Timer {
    fn new() -> Timer {
        Timer {
            last_timestamp: Instant::now(),
        }
    }

    fn measure(&mut self, msg: &str) {
        let new_timestamp = Instant::now();
        let seconds = duration_as_seconds(new_timestamp - self.last_timestamp);
        println!("[TIMER] ({}) took {}s", msg, seconds);
        self.last_timestamp = new_timestamp;
    }
}

fn main() {
    let mut args = args_os();
    args.next();
    let path = args.next().unwrap();
    let fineness = args
        .next()
        .unwrap()
        .to_string_lossy()
        .parse::<u32>()
        .unwrap();

    let mut timer = Timer::new();
    let mut img = image::open(&Path::new(&path)).unwrap().to_rgb();
    let point_count = ((img.width() * img.height()) / fineness) as usize;

    println!("Point count: {}", point_count);
    timer.measure("Opening image file");

    let diff_img = compute_differential_image(&img);
    timer.measure("Computing difference");

    let initial_points = generate_initial_points(&diff_img, point_count, img.width(), img.height());
    timer.measure("Generating initial points");

    let tris = delaunay::triangulate(
        &initial_points,
        (0.0, 0.0),
        (img.width() as f64, img.height() as f64),
    );
    timer.measure("Calculating delaunay triangulation");

    generate_delaunay_image(&mut img, &tris);
    timer.measure("Generating delaunay image");

    let new_path = String::from(path.to_string_lossy()) + ".cellied.png";
    img.save(OsString::from(new_path)).unwrap();
    timer.measure("Saving");
}
