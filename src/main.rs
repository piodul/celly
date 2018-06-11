extern crate image;
extern crate rayon;

mod float_ord;
mod tree_2d;
mod halton;
mod delaunay;

use std::env::args_os;
use std::cmp;
use std::path::Path;
use std::ffi::OsString;
use std::time::{Instant, Duration};
use image::Pixel;
use rayon::prelude::*;
use float_ord::FloatOrd;

use tree_2d::{Point2D, Tree2D, Coord, NearestNeighbor2D, StupidFind};
use halton::HaltonSequence;

fn convert_to_luma(color: &image::Rgb<u8>) -> Coord {
    0.2126 * color.data[0] as Coord + 0.7152 * color.data[1] as Coord + 0.0722 * color.data[2] as Coord
}

fn generate_initial_points(diff_img: &Vec<Coord>, n: usize, width: u32, height: u32) -> Vec<Point2D> {
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

fn generate_celly_image<T: NearestNeighbor2D + Sync>(img: &mut image::RgbImage, points: Vec<Point2D>) {
    let pt1 = Instant::now();
    let plen = points.len();
    let tree = T::new_from_points(points);
    let mut acc_colors = vec![(0.0, 0.0, 0.0, 0); plen];
    let mut fy = 0.5;

    let im_width = img.width();
    let im_height = img.height();

    // let mut idx_cache = Vec::with_capacity((img.width() * img.height()) as usize);
    let mut idx_cache = vec![0; (im_width * im_height) as usize];
    idx_cache.par_iter_mut().enumerate().for_each(|(id, cell)| {
        let x = id as i32 % im_width as i32;
        let y = id as i32 / im_width as i32;
        let fx = x as f64 + 0.5;
        let fy = y as f64 + 0.5;
        let (closest_idx, cnt) = tree.find_closest((fx, fy));
        *cell = closest_idx;
    });

    let mut pix_id = 0;
    for y in 0..im_height {
        for x in 0..im_width {
            let px = img.get_pixel(x, y);
            // let (closest_idx, cnt) = tree.find_closest((fx, fy));
            let closest_idx = idx_cache[pix_id];
            let acc = &mut acc_colors[closest_idx as usize];
            acc.0 += px.data[0] as Coord;
            acc.1 += px.data[1] as Coord;
            acc.2 += px.data[2] as Coord;
            acc.3 += 1;

            pix_id += 1;
        }
    }

    for acc in acc_colors.iter_mut() {
        let recip = 1.0 / acc.3 as Coord;
        acc.0 *= recip;
        acc.1 *= recip;
        acc.2 *= recip;
    }

    let pt2 = Instant::now();

    fy = 0.5;
    let mut pix_id = 0;
    for y in 0..im_height {
        let mut fx = 0.5;
        for x in 0..im_width {
            let closest_idx = idx_cache[pix_id];
            pix_id += 1;
            // let closest_idx = tree.find_closest((fx, fy)).0;
            let acc = &acc_colors[closest_idx as usize];
            img.put_pixel(x, y, image::Rgb {
                data: [acc.0 as u8, acc.1 as u8, acc.2 as u8]
            });

            fx += 1.0;
        }
        fy += 1.0;
    }

    let pt3 = Instant::now();
    println!("    Collecting sums took {}s", duration_as_seconds(pt2 - pt1));
    println!("    Setting colors took {}s", duration_as_seconds(pt3 - pt2));

    // let wastefulness = total_visited_nodes as f64 / (total_lookups as f64 * acc_colors.len() as f64);
    // println!("Wastefulness: {}", wastefulness);
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
    let fineness = args.next().unwrap().to_string_lossy().parse::<u32>().unwrap();

    let mut timer = Timer::new();
    let mut img = image::open(&Path::new(&path)).unwrap().to_rgb();
    let point_count = ((img.width() * img.height()) / fineness) as usize;

    println!("Point count: {}", point_count);
    timer.measure("Opening image file");

    let diff_img = compute_differential_image(&img);
    timer.measure("Computing difference");

    let initial_points = generate_initial_points(&diff_img, point_count, img.width(), img.height());
    timer.measure("Generating initial points");

    // let points = migrate_points(&img, &diff_img, initial_points, img.width(), img.height(), 10);
    delaunay::triangulate(&initial_points, (0.0, 0.0), (1920.0, 1080.0));
    timer.measure("Calculating delaunay triangulation");

    generate_celly_image::<Tree2D>(&mut img, initial_points);
    timer.measure("Generating celly image");

    let new_path = String::from(path.to_string_lossy()) + ".cellied.png";
    img.save(OsString::from(new_path)).unwrap();
    timer.measure("Saving");
}
