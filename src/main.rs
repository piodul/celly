extern crate image;
extern crate rayon;

mod common_geometry;
mod delaunay;
mod float_ord;
mod halton;
mod rasterization;
mod tree_2d;

use crate::common_geometry::{BarycentricConverter, Triangle2D};
use crate::float_ord::FloatOrd;
use image::Pixel;
use rayon::prelude::*;
use std::cmp;
use std::collections::HashMap;
use std::env::args_os;
use std::ffi::OsString;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::halton::HaltonSequence;
use crate::tree_2d::{Coord, Point2D};

#[derive(Clone, Default)]
struct TriangleColorConfiguration {
    pub colors: [(f64, f64, f64); 3],
    pub weights: [f64; 3],
}

fn convert_to_luma(color: &image::Rgb<u8>) -> Coord {
    0.2126 * color.channels()[0] as Coord
        + 0.7152 * color.channels()[1] as Coord
        + 0.0722 * color.channels()[2] as Coord
}

fn generate_initial_points(diff_img: &[Coord], n: usize, width: u32, height: u32) -> Vec<Point2D> {
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

fn generate_delaunay_image(
    timer: &mut Timer,
    img: &mut image::RgbImage,
    triangulation: &Vec<Triangle2D>,
) {
    let im_width = img.width() as usize;
    let im_height = img.height() as usize;

    let events = rasterization::prepare_aa_events(triangulation);
    timer.measure("Preparing events");

    // Calculate barycentric converters and colors
    let bary_converters = triangulation
        .iter()
        .map(BarycentricConverter::from_triangle)
        .collect::<Vec<_>>();

    let computed = rasterization::rasterize(&events, im_width, im_height);
    timer.measure("Computing coverage info for rasterization");

    let mut ccinfo = {
        let ccinfos = computed
            .chunks()
            .par_iter()
            .enumerate()
            .map(|(id, chunk)| {
                let mut ccinfo: HashMap<usize, TriangleColorConfiguration> = HashMap::new();

                let y_start = id * computed.rows_per_chunk();
                let y_end =
                    std::cmp::min(y_start + computed.rows_per_chunk(), img.height() as usize);
                chunk.replay(
                    y_start,
                    y_end,
                    img.width() as usize,
                    |x: u32, y: u32, coverage: &[(usize, f64)]| {
                        for (tri_id, factor) in coverage.iter().cloned() {
                            let cc = &mut ccinfo.entry(tri_id).or_insert_with(Default::default);
                            let point = img.get_pixel(x, y);
                            let (mut a, mut b, mut c) = bary_converters[tri_id]
                                .convert_to_barycentric((x as f64 + 0.5, y as f64 + 0.5));
                            a *= factor;
                            b *= factor;
                            c *= factor;
                            cc.colors[0].0 += a * point.channels()[0] as f64;
                            cc.colors[0].1 += a * point.channels()[1] as f64;
                            cc.colors[0].2 += a * point.channels()[2] as f64;
                            cc.colors[1].0 += b * point.channels()[0] as f64;
                            cc.colors[1].1 += b * point.channels()[1] as f64;
                            cc.colors[1].2 += b * point.channels()[2] as f64;
                            cc.colors[2].0 += c * point.channels()[0] as f64;
                            cc.colors[2].1 += c * point.channels()[1] as f64;
                            cc.colors[2].2 += c * point.channels()[2] as f64;
                            cc.weights[0] += a;
                            cc.weights[1] += b;
                            cc.weights[2] += c;
                        }
                    },
                );

                ccinfo
            })
            .collect::<Vec<_>>();

        timer.measure("Computing triangle colors");

        let mut target_ccinfo = vec![TriangleColorConfiguration::default(); triangulation.len()];

        for ccinfo in ccinfos {
            for (tri_id, ccinfo) in ccinfo {
                let cc = &mut target_ccinfo[tri_id];
                cc.colors[0].0 += ccinfo.colors[0].0;
                cc.colors[0].1 += ccinfo.colors[0].1;
                cc.colors[0].2 += ccinfo.colors[0].2;
                cc.colors[1].0 += ccinfo.colors[1].0;
                cc.colors[1].1 += ccinfo.colors[1].1;
                cc.colors[1].2 += ccinfo.colors[1].2;
                cc.colors[2].0 += ccinfo.colors[2].0;
                cc.colors[2].1 += ccinfo.colors[2].1;
                cc.colors[2].2 += ccinfo.colors[2].2;
                cc.weights[0] += ccinfo.weights[0];
                cc.weights[1] += ccinfo.weights[1];
                cc.weights[2] += ccinfo.weights[2];
            }
        }

        timer.measure("Gathering triangle colors");

        target_ccinfo
    };

    // Scale the colors
    for cc in ccinfo.iter_mut() {
        if cc.weights[0] != 0.0 {
            cc.colors[0].0 /= cc.weights[0];
            cc.colors[0].1 /= cc.weights[0];
            cc.colors[0].2 /= cc.weights[0];
        }
        if cc.weights[1] != 0.0 {
            cc.colors[1].0 /= cc.weights[1];
            cc.colors[1].1 /= cc.weights[1];
            cc.colors[1].2 /= cc.weights[1];
        }
        if cc.weights[2] != 0.0 {
            cc.colors[2].0 /= cc.weights[2];
            cc.colors[2].1 /= cc.weights[2];
            cc.colors[2].2 /= cc.weights[2];
        }
    }

    let mut pixel_id = 0;
    computed.replay(|x: u32, y: u32, coverage: &[(usize, f64)]| {
        let pixel = {
            // let csum = coverage.iter().map(|(_, c)| *c).sum::<f64>();
            // if csum < 0.99 || csum > 1.01 {
            //     img.put_pixel(x, y, image::Rgb([255, 0, 0]));
            //     pixel_id += 1;
            //     return;
            // }

            let mut color = [0.0; 3];
            for (id, factor) in coverage.iter().cloned() {
                let cc = &ccinfo[id];
                let (a, b, c) =
                    bary_converters[id].convert_to_barycentric((x as f64 + 0.5, y as f64 + 0.5));
                color[0] += (a * cc.colors[0].0 + b * cc.colors[1].0 + c * cc.colors[2].0) * factor;
                color[1] += (a * cc.colors[0].1 + b * cc.colors[1].1 + c * cc.colors[2].1) * factor;
                color[2] += (a * cc.colors[0].2 + b * cc.colors[1].2 + c * cc.colors[2].2) * factor;
            }
            // The factors/weights are assumed to sum up to 1.0
            image::Rgb([color[0] as u8, color[1] as u8, color[2] as u8])
        };
        // let pixel = if coverage.len() == 1 {
        //     image::Rgb([0, 255, 0])
        // } else {
        //     image::Rgb([255, 0, 0])
        // };
        // let pixel = image::Rgb([(10 * coverage.len()) as u8, 0, 0]);
        img.put_pixel(x, y, pixel);
        pixel_id += 1;
    });
    timer.measure("Generating image");
}

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
    let mut img = image::open(Path::new(&path)).unwrap().to_rgb8();
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

    generate_delaunay_image(&mut timer, &mut img, &tris);

    let new_path = String::from(path.to_string_lossy()) + ".cellied.png";
    img.save(OsString::from(new_path)).unwrap();
    timer.measure("Saving");
}
