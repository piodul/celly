extern crate image;
extern crate rayon;

mod float_ord;
mod tree_2d;
mod halton;

use std::env::args_os;
use std::cmp;
use std::path::Path;
use std::ffi::OsString;
use std::time::{Instant, Duration};
use image::Pixel;
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
    let mut ret = Vec::with_capacity((img.width() * img.height()) as usize);
    for y in 0..img.height() {
        for x in 0..img.width() {
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
            let v = diff_x * diff_x + diff_y * diff_y;
            ret.push(v);
        }
    }

    // Blur the image
    const BLUR_RADIUS: i32 = 5;
    let mut ret2 = Vec::with_capacity((img.width() * img.height()) as usize);
    for y in 0..img.height() as i32 {
        for x in 0..img.width() as i32 {
            let mut acc = 0.0;
            for by in -BLUR_RADIUS..BLUR_RADIUS + 1 {
                let oy = cmp::max(0, cmp::min(img.height() as i32 - 1, y + by));
                for bx in -BLUR_RADIUS..BLUR_RADIUS + 1 {
                    let ox = cmp::max(0, cmp::min(img.width() as i32 - 1, x + bx));
                    let idx = oy as u32 * img.width() + ox as u32;
                    acc += ret[idx as usize];
                }
            }
            ret2.push(acc);
        }
    }
    ret2
}

fn migrate_points(
    img: &image::RgbImage,
    differential: &Vec<Coord>,
    mut points: Vec<Point2D>,
    width: u32,
    height: u32,
    max_iters: u32,
) -> Vec<Point2D> {
    // println!("{:?}", points);
    let mut point_weights = Vec::with_capacity(points.len());

    let mut niters = 0;

    let mut previous_diff = 1.0;
    while previous_diff > 0.0 && niters < max_iters {
        point_weights.clear();
        point_weights.resize(points.len(), ((0.0, 0.0), 0.0));

        let mut tree = Tree2D::new_from_points(points.clone());

        let mut idx = 0;
        let mut fy = 0.5;
        for _ in 0..height {
            let mut fx = 0.5;
            for _ in 0..width {
                let (closest_idx, _) = tree.find_closest((fx, fy));
                let weight = differential[idx];
                (point_weights[closest_idx as usize].0).0 += fx * weight;
                (point_weights[closest_idx as usize].0).1 += fy * weight;
                point_weights[closest_idx as usize].1 += weight;

                fx += 1.0;
                idx += 1;
            }
            fy += 1.0;
        }

        previous_diff = 0.0;
        let mut new_points = Vec::new();
        for (id, (pt, w)) in point_weights.iter().enumerate() {
            if *w > 0.0 {
                let ptx = pt.0 / w;
                let pty = pt.1 / w;
                new_points.push((ptx, pty));
                let oldpt = points[id];
                let d = (oldpt.0 - ptx).abs() + (oldpt.1 - pty).abs();
                if previous_diff < d {
                    previous_diff = d;
                }
            }
        }

        // let mut celly = img.clone();
        // generate_celly_image(&mut celly, new_points.clone());
        // let new_path = format!("/tmp/cell-{:04}.png", niters);
        // celly.save(OsString::from(new_path)).unwrap();
        niters += 1;

        std::mem::swap(&mut points, &mut new_points);

        println!("Previous diff: {}, points remaining: {}", previous_diff, points.len());
    }

    points
}

fn generate_celly_image<T: NearestNeighbor2D>(img: &mut image::RgbImage, points: Vec<Point2D>) {
    let mut total_visited_nodes = 0;
    let mut total_lookups = 0;

    let mut acc_colors = vec![(0.0, 0.0, 0.0, 0); points.len()];
    let mut tree = T::new_from_points(points);
    let mut fy = 0.5;
    for y in 0..img.height() {
        let mut fx = 0.5;
        for x in 0..img.width() {
            let px = img.get_pixel(x, y);
            let (closest_idx, cnt) = tree.find_closest((fx, fy));
            let acc = &mut acc_colors[closest_idx as usize];
            total_visited_nodes += cnt;
            total_lookups += 1;
            acc.0 += px.data[0] as Coord;
            acc.1 += px.data[1] as Coord;
            acc.2 += px.data[2] as Coord;
            acc.3 += 1;

            fx += 1.0;
        }
        fy += 1.0;
    }

    for acc in acc_colors.iter_mut() {
        let recip = 1.0 / acc.3 as Coord;
        acc.0 *= recip;
        acc.1 *= recip;
        acc.2 *= recip;
    }

    fy = 0.5;
    for y in 0..img.height() {
        let mut fx = 0.5;
        for x in 0..img.width() {
            let closest_idx = tree.find_closest((fx, fy)).0;
            let acc = &acc_colors[closest_idx as usize];
            img.put_pixel(x, y, image::Rgb {
                data: [acc.0 as u8, acc.1 as u8, acc.2 as u8]
            });

            fx += 1.0;
        }
        fy += 1.0;
    }

    let wastefulness = (total_visited_nodes as f64 / (total_lookups as f64 * acc_colors.len() as f64));
    println!("Wastefulness: {}", wastefulness);
}

// fn draw_distribution(points: &Vec<Point2D>) -> image::RgbImage {
//     let mut img =
// }

fn duration_as_seconds(d: Duration) -> f64 {
    d.as_secs() as f64 + d.subsec_micros() as f64 / 1e6
}

fn main() {
    let mut args = args_os();
    args.next();
    let path = args.next().unwrap();
    let fineness = args.next().unwrap().to_string_lossy().parse::<u32>().unwrap();

    let tpt1 = Instant::now();
    let mut img = image::open(&Path::new(&path)).unwrap().to_rgb();
    let point_count = ((img.width() * img.height()) / fineness) as usize;

    let tpt2 = Instant::now();
    println!("Point count: {}", point_count);
    println!("  Opening took {}s", duration_as_seconds(tpt2 - tpt1));
    let diff_img = compute_differential_image(&img);

    let tpt3 = Instant::now();
    println!("  Computing difference took {}s", duration_as_seconds(tpt3 - tpt2));
    let initial_points = generate_initial_points(&diff_img, point_count, img.width(), img.height());
    // let points = migrate_points(&img, &diff_img, initial_points, img.width(), img.height(), 10);

    let tpt4 = Instant::now();
    println!("  Generating initial points took {}s", duration_as_seconds(tpt4 - tpt3));
    generate_celly_image::<Tree2D>(&mut img, initial_points);

    let tpt5 = Instant::now();
    println!("  Generating celly image took {}s", duration_as_seconds(tpt5 - tpt4));
    let new_path = String::from(path.to_string_lossy()) + ".cellied.png";
    img.save(OsString::from(new_path)).unwrap();

    let tpt6 = Instant::now();
    println!("  Saving took {}s", duration_as_seconds(tpt6 - tpt5));
}
