use crate::float_ord::FloatOrd;

pub type Coord = f64;
pub type Point2D = (Coord, Coord);

const SWITCH_TO_LINEAR: u32 = 12;

pub trait NearestNeighbor2D {
    fn new_from_points(points: Vec<Point2D>) -> Self;
    fn find_closest(&self, point: Point2D) -> (u32, usize);
    fn get_nth_point_position(&self, n: u32) -> Point2D;
}

struct Tree2DNode {
    pub point: Point2D,
    pub left: u32,
    pub right: u32,
}

pub struct Tree2D {
    nodes: Vec<Tree2DNode>,
}

fn distance_squared(p1: Point2D, p2: Point2D) -> Coord {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    dx * dx + dy * dy
}

impl Tree2D {
    fn new_inner_from_points(&mut self, depth: u32, points: &mut [Point2D]) -> u32 {
        let idx = self.nodes.len() as u32;

        if points.len() <= SWITCH_TO_LINEAR as usize {
            for pt in points.iter() {
                self.nodes.push(Tree2DNode {
                    point: *pt,
                    left: 0,
                    right: 0,
                })
            }
            return idx;
        }

        if depth % 2 == 0 {
            points.sort_unstable_by_key(|p| FloatOrd(p.0));
        } else {
            points.sort_unstable_by_key(|p| FloatOrd(p.1));
        }

        let median_index = points.len() / 2;
        let plen = points.len();

        self.nodes.push(Tree2DNode {
            point: points[median_index],
            left: 0,
            right: 0,
        });

        self.nodes[idx as usize].left =
            self.new_inner_from_points(depth + 1, &mut points[0..median_index]);
        self.nodes[idx as usize].right =
            self.new_inner_from_points(depth + 1, &mut points[median_index + 1..plen]);
        idx
    }

    fn find_closest_inner(
        &self,
        closest: &mut (u32, Coord),
        depth: u32,
        curr_range: u32,
        idx: u32,
        pt: Point2D,
    ) -> usize {
        if curr_range <= SWITCH_TO_LINEAR {
            for (id, node) in self.nodes[idx as usize..(idx + curr_range) as usize]
                .iter()
                .enumerate()
            {
                let new_distance = distance_squared(pt, node.point);
                if closest.1 > new_distance {
                    closest.0 = idx + id as u32;
                    closest.1 = new_distance;
                }
            }

            return curr_range as usize;
        }

        let node = &self.nodes[idx as usize];
        let new_distance = distance_squared(pt, node.point);
        if closest.1 > new_distance {
            closest.0 = idx;
            closest.1 = new_distance;
        }

        let left_len = curr_range / 2;
        let right_len = curr_range - (curr_range / 2) - 1;

        let is_even = depth % 2 == 0;
        let (first_node, first_len, next_node, next_len) =
            if (is_even && pt.0 < node.point.0) || (!is_even && pt.1 < node.point.1) {
                (node.left, left_len, node.right, right_len)
            } else {
                (node.right, right_len, node.left, left_len)
            };

        let mut visited_count = 1;
        visited_count += self.find_closest_inner(closest, depth + 1, first_len, first_node, pt);

        let distance = if is_even {
            pt.0 - node.point.0
        } else {
            pt.1 - node.point.1
        };
        if distance * distance < closest.1 {
            visited_count += self.find_closest_inner(closest, depth + 1, next_len, next_node, pt);
        }

        visited_count
    }
}

impl NearestNeighbor2D for Tree2D {
    fn new_from_points(mut points: Vec<Point2D>) -> Self {
        assert!(!points.is_empty());

        let mut ret = Tree2D {
            nodes: Vec::with_capacity(points.len()),
        };
        ret.new_inner_from_points(0, points.as_mut_slice());
        ret
    }

    fn find_closest(&self, point: Point2D) -> (u32, usize) {
        let mut p = (0, distance_squared(point, self.nodes[0].point));
        let cnt = self.find_closest_inner(&mut p, 0, self.nodes.len() as u32, 0, point);
        (p.0, cnt)
    }

    fn get_nth_point_position(&self, n: u32) -> Point2D {
        self.nodes[n as usize - 1].point
    }
}

pub struct StupidFind {
    points: Vec<Point2D>,
}

impl NearestNeighbor2D for StupidFind {
    fn new_from_points(points: Vec<Point2D>) -> Self {
        StupidFind { points }
    }

    fn find_closest(&self, point: Point2D) -> (u32, usize) {
        let mut closest_id = 0;
        let mut closest_dist = distance_squared(self.points[0], point);
        for (id, value) in self.points.iter().enumerate() {
            let new_dist = distance_squared(*value, point);
            if closest_dist > new_dist {
                closest_dist = new_dist;
                closest_id = id;
            }
        }

        (closest_id as u32, self.points.len())
    }

    fn get_nth_point_position(&self, n: u32) -> Point2D {
        self.points[n as usize]
    }
}
