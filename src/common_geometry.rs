pub type Coord = f64;
pub type Point2D = (Coord, Coord);
pub type Triangle2D = [Point2D; 3];

pub fn det2(pt00: Coord, pt01: Coord, pt10: Coord, pt11: Coord) -> Coord {
    pt00 * pt11 - pt01 * pt10
}

pub fn point_in_triangle(point: &Point2D, tri: &Triangle2D) -> bool {
    let check1 = is_left_from_segment(tri[0], tri[1], *point);
    let check2 = is_left_from_segment(tri[1], tri[2], *point);
    let check3 = is_left_from_segment(tri[2], tri[0], *point);
    (check1 == check2) && (check2 == check3)
}

pub fn is_ccw(tri: &Triangle2D) -> bool {
    goes_left(tri[0], tri[1], tri[2])
}

pub fn is_left_from_segment((p1x, p1y): Point2D, (p2x, p2y): Point2D, (qx, qy): Point2D) -> bool {
    det2(p2x - p1x, p2y - p1y, qx - p1x, qy - p1y) >= 0.0
}

pub fn goes_left((ox, oy): Point2D, (ux, uy): Point2D, (vx, vy): Point2D) -> bool {
    (ux - ox) * (vy - oy) - (uy - oy) * (vx - ux) >= 0.0
}
