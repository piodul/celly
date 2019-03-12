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

pub struct BarycentricConverter {
    mat: [[Coord; 2]; 2],
    v: Point2D,
}

impl BarycentricConverter {
    pub fn from_triangle(t: &Triangle2D) -> BarycentricConverter {
        let o0 = t[0];
        let v0 = (t[1].0 - t[0].0, t[1].1 - t[0].1);
        let v1 = (t[2].0 - t[0].0, t[2].1 - t[0].1);
        let invd = 1.0 / det2(v0.0, v0.1, v1.0, v1.1);
        BarycentricConverter {
            mat: [[v1.1 * invd, -v1.0 * invd], [-v0.1 * invd, v0.0 * invd]],
            v: o0,
        }
    }

    pub fn convert_to_barycentric(&self, (x, y): Point2D) -> (Coord, Coord, Coord) {
        let dx = x - self.v.0;
        let dy = y - self.v.1;
        let u = self.mat[0][0] * dx + self.mat[0][1] * dy;
        let v = self.mat[1][0] * dx + self.mat[1][1] * dy;
        (1.0 - u - v, u, v)
    }
}
