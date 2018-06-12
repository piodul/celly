pub type Coord = f64;
pub type Point2D = (Coord, Coord);
pub type Triangle2D = [Point2D; 3];

pub fn det2(
    pt00: Coord, pt01: Coord,
    pt10: Coord, pt11: Coord,
) -> Coord {
    pt00 * pt11 - pt01 * pt10
}

pub fn det3(
    p00: Coord, p01: Coord, p02: Coord,
    p10: Coord, p11: Coord, p12: Coord,
    p20: Coord, p21: Coord, p22: Coord,
) -> Coord {
    let d0 = p11 * p22 - p21 * p12;
    let d1 = p21 * p02 - p01 * p22;
    let d2 = p01 * p12 - p11 * p02;
    p00 * d0 + p10 * d1 + p20 * d2
}

pub fn point_in_triangle(point: &Point2D, tri: &Triangle2D) -> bool {
    let check1 = is_left_from_segment(tri[0], tri[1], *point);
    let check2 = is_left_from_segment(tri[1], tri[2], *point);
    let check3 = is_left_from_segment(tri[2], tri[0], *point);
    (check1 == check2) && (check2 == check3)
}

pub fn distance_sq((x1, y1): Point2D, (x2, y2): Point2D) -> Coord {
    let dx = x1 - x2;
    let dy = y1 - y2;
    dx * dx + dy * dy
}

pub fn is_ccw(tri: &Triangle2D) -> bool {
    goes_left(tri[0], tri[1], tri[2])
}

pub fn is_left_from_segment(
    (p1x, p1y): Point2D, (p2x, p2y): Point2D,
    (qx, qy): Point2D,
) -> bool {
    det2(p2x - p1x, p2y - p1y, qx - p1x, qy - p1y) >= 0.0
}

pub fn goes_left((ox, oy): Point2D, (ux, uy): Point2D, (vx, vy): Point2D) -> bool {
    (ux - ox) * (vy - oy) - (uy - oy) * (vx - ux) >= 0.0
}