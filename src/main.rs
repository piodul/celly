extern crate image;
extern crate rayon;

mod float_ord;
mod tree_2d;

use std::env::args_os;

fn main() {
    let mut args = args_os();
    args.next();
    let path = args.next().unwrap();



    println!("Hello, world!");
}
