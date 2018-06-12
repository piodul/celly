use std::f64;
use std::collections::{HashSet, VecDeque};
use std::mem;
use std::ops::{Deref, DerefMut};

use std::fs::File;
use std::io::BufWriter;
use std::io;
use std::io::prelude::*;

use common_geometry::*;
use float_ord::FloatOrd;

#[derive(Debug)]
struct TriWithNbs {
    pub pts: Triangle2D,
    pub nbs: [usize; 3],
}

struct TriangleNode {
    split_point: Point2D,
    children: [Option<Box<TriangleNode>>; 3],
}

impl TriangleNode {
    fn new(split_point: Point2D) -> TriangleNode {
        TriangleNode {
            split_point,
            children: [None, None, None],
        }
    }

    fn insert_point(&mut self, mut tri: Triangle2D, point: Point2D) {
        let mut node = self;
        loop {
            //      0
            //     /|\     That was supposed to be three triangles
            //    / | \    ...close enough
            //   / _X_ \
            //  /_-   -_\
            // 1---------2
            let id = if is_left_from_segment(node.split_point, tri[0], point) {
                // Is in triangle opposite to 2 or 0
                match is_left_from_segment(node.split_point, tri[1], point) {
                    true => 0,
                    false => 2,
                }
            }
            else {
                // Is in triangle opposite to 1 or 0
                match is_left_from_segment(node.split_point, tri[2], point) {
                    true => 1,
                    false => 0,
                }
            };

            if node.children[id].is_none() {
                node.children[id] = Some(Box::new(TriangleNode::new(point)));
                return;
            }

            tri[id] = node.split_point;

            // Workaround for the borrow checker
            let tmp = node;
            node = tmp.children[id].as_mut().unwrap().deref_mut();
        }
    }

    // Dumps triangles from the structure.
    // Returns indexes of triangles that sit at the edges of the bigger tri.
    // Triangles are indexed from 1, 0 is reserved for flagging that
    // there is no neighbor
    fn dump_triangles(
        o_node: Option<&TriangleNode>,
        tri: Triangle2D,
        v: &mut Vec<TriWithNbs>,
    ) -> [usize; 3] {
        match o_node {
            Some(node) => {
                // Compute triangles in lower nodes
                let e0 = TriangleNode::dump_triangles(
                    node.children[0].as_ref().map(|p| p.deref()),
                    [node.split_point, tri[1], tri[2]],
                    v,
                );
                let e1 = TriangleNode::dump_triangles(
                    node.children[1].as_ref().map(|p| p.deref()),
                    [tri[0], node.split_point, tri[2]],
                    v,
                );
                let e2 = TriangleNode::dump_triangles(
                    node.children[2].as_ref().map(|p| p.deref()),
                    [tri[0], tri[1], node.split_point],
                    v,
                );

                // Connect edge triangles
                v[e0[1] - 1].nbs[1] = e1[0];
                v[e1[0] - 1].nbs[0] = e0[1];

                v[e1[2] - 1].nbs[2] = e2[1];
                v[e2[1] - 1].nbs[1] = e1[2];

                v[e2[0] - 1].nbs[0] = e0[2];
                v[e0[2] - 1].nbs[2] = e2[0];

                [e0[0], e1[1], e2[2]]
            },
            None => {
                v.push(TriWithNbs {
                    pts: tri,
                    nbs: [0; 3],
                });
                [v.len(); 3]
            },
        }
    }
}

struct TriangleTree {
    core_triangle: Triangle2D,
    root: Option<TriangleNode>,
}

impl TriangleTree {
    fn new(mut core_triangle: [Point2D; 3]) -> TriangleTree {
        if !is_ccw(&core_triangle) {
            core_triangle.swap(1, 2);
        }
        TriangleTree {
            core_triangle,
            root: None,
        }
    }

    fn insert_point(&mut self, pt: Point2D) -> bool {
        if !point_in_triangle(&pt, &self.core_triangle) {
            return false;
        }

        if let Some(node) = &mut self.root {
            node.insert_point(self.core_triangle, pt);
            return true;
        }

        self.root = Some(TriangleNode::new(pt));
        true
    }

    fn dump_triangles(&self) -> Vec<TriWithNbs> {
        let mut v = Vec::new();
        TriangleNode::dump_triangles(
            self.root.as_ref(),
            self.core_triangle,
            &mut v,
        );
        v
    }
}

// Idea taken from s-hull implementation
fn cline_renka_test(
    (ax, ay): Point2D,
    (bx, by): Point2D,
    (cx, cy): Point2D,
    (dx, dy): Point2D,
) -> bool {
    //       v1
    //       ->
    //      A--D
    // v2 | | /| ^
    //    v |/ | | v3
    //      B--C
    //       <-
    //       v4
    let (v1x, v1y) = (dx - ax, dy - ay);
    let (v2x, v2y) = (bx - ax, by - ay);
    let (v3x, v3y) = (dx - cx, dy - cy);
    let (v4x, v4y) = (bx - cx, by - cy);
    let cosA = v1x * v2x + v1y * v2y;
    let cosD = v3x * v4x + v3y * v4y;

    if cosA < 0.0 && cosD < 0.0 {
        return true;
    }
    if cosA > 0.0 && cosD > 0.0 {
        return false;
    }

    let sinA = det2(v1x, v1y, v2x, v2y).abs();
    let sinD = det2(v3x, v3y, v4x, v4y).abs();

    cosA * sinD + cosD * sinA < 0.0
}

fn dump_triangles(file_id: usize, tris: &Vec<TriWithNbs>) -> io::Result<()> {
    let file = File::create(format!("dump_{:04}.txt", file_id))?;
    let mut buf_writer = BufWriter::new(file);
    buf_writer.write_fmt(format_args!("{}\n", tris.len()))?;
    for tri in tris.iter() {
        buf_writer.write_fmt(format_args!(
            "{} {} {} {} {} {}\n",
            tri.pts[0].0, tri.pts[0].1,
            tri.pts[1].0, tri.pts[1].1,
            tri.pts[2].0, tri.pts[2].1,
        ));
    }
    Ok(())
}

fn flip_until_delaunay(tris: &mut Vec<TriWithNbs>) {
    let mut queue = VecDeque::with_capacity(tris.len());
    for i in 0..tris.len() {
        queue.push_back(i);
    }

    let mut iterations = 0;

    let get_two_other_ids = |x| {
        match x {
            0 => (1, 2),
            1 => (2, 0),
            2 => (0, 1),
            _ => unreachable!(),
        }
    };

    // tid is indexed from 0, while neighbors are indexed from 1
    while let Some(tid) = queue.pop_front() {
        iterations += 1;
        for nb_num in 0..3 {
            let nb_id = match tris[tid].nbs[nb_num] {
                0 => continue, // No neighbor at this side
                n => n - 1,
            };

            // Get number of the neighbor's side that is shared with
            // current triangle
            let nb_nb_num = tris[nb_id].nbs.iter()
                .position(|id| *id == tid + 1).unwrap();
            let a_num = nb_num;
            let (b_num, x_num) = get_two_other_ids(nb_num);
            let c_num = nb_nb_num;
            let (d_num, y_num) = get_two_other_ids(nb_nb_num);

            if cline_renka_test(
                tris[tid].pts[a_num],
                tris[tid].pts[b_num],
                tris[nb_id].pts[c_num],
                tris[nb_id].pts[d_num],
            ) {
                // println!("Flipping!");
                // +--A--+      +--A--+
                //  \1|\1|       \1|\2|
                //   \| \|        \|\\|
                //    B--D    ->   B  D
                //    |\ |\        |\\|\
                //    |2\|2\       |1\|2\
                //    +--C--+      +--C--+

                // Update references in neighbors
                let b_nb = tris[tid].nbs[b_num];
                if b_nb != 0 {
                    let slot_to_update = tris[b_nb - 1].nbs.iter()
                        .position(|id| *id == tid + 1).unwrap();
                    tris[b_nb - 1].nbs[slot_to_update] = nb_id + 1;
                }

                let b_nb = tris[nb_id].nbs[d_num];
                if b_nb != 0 {
                    let slot_to_update = tris[b_nb - 1].nbs.iter()
                        .position(|id| *id == nb_id + 1).unwrap();
                    tris[b_nb - 1].nbs[slot_to_update] = tid + 1;
                }

                // Update references to our neighbors
                let old_nbs1 = tris[tid].nbs;
                let old_nbs2 = tris[nb_id].nbs;
                tris[tid].nbs[a_num] = old_nbs2[d_num];
                tris[tid].nbs[b_num] = nb_id + 1;
                tris[nb_id].nbs[c_num] = old_nbs1[b_num];
                tris[nb_id].nbs[d_num] = tid + 1;

                // Update points
                tris[tid].pts[x_num] = tris[nb_id].pts[c_num];
                tris[nb_id].pts[y_num] = tris[tid].pts[a_num];

                // Re-check me and the neighbor later
                queue.push_back(tid);
                queue.push_back(nb_id);

                // Don't flip further
                break;
            }
        }
    }

    println!("Took {} iterations", iterations);
}

pub fn triangulate(
    points: &Vec<Point2D>,
    (ax, ay): Point2D,
    (bx, by): Point2D,
) -> Vec<Triangle2D> {
    // Calculate a triangle that encompasses whole rectangle
    let base_tri = [
        (ax - (by - ay), by),
        (bx + (by - ay), by),
        ((ax + bx) / 2.0, ax - (bx - ax) / 2.0),
    ];

    // Populate structure with points
    let mut tt = TriangleTree::new(base_tri);
    for pt in points.iter() {
        tt.insert_point(*pt);
    }

    // Dump triangles from the structure
    let mut v = tt.dump_triangles();

    // Make a delaunay triangulation from it
    flip_until_delaunay(&mut v);

    // Dump triangles to file (for debug purposes)
    // dump_triangles(0, &v).unwrap();

    v.into_iter().map(|t| t.pts).collect()
}
