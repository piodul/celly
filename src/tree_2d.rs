use float_ord::FloatOrd;

type Point2D = (f64, f64);

struct Tree2DNode<V> {
    pub point: Point2D,
    pub value: V,
    pub left: usize,
    pub right: usize,
}

pub struct Tree2D<V> {
    nodes: Vec<Tree2DNode<V>>,
}

fn make_subtree<V: Copy>(depth: u32, mut points: &mut [(Point2D, V)]) -> Option<Box<Tree2DNode<V>>> {
    if points.is_empty() {
        return None
    }

    if depth % 2 == 0 {
        points.sort_unstable_by_key(|p| FloatOrd((p.0).0));
    }
    else {
        points.sort_unstable_by_key(|p| FloatOrd((p.0).1));
    }

    let median_index = points.len() / 2;
    let plen = points.len();
    let left = make_subtree(depth + 1, &mut points[0..median_index]);
    let right = make_subtree(depth + 1, &mut points[median_index + 1..plen]);
    Some(Box::new(Tree2DNode {
        point: points[median_index].0,
        value: points[median_index].1,
        left, right,
    }))
}

fn find_closest<V>(depth: u32, node: &Tree2DNode<V>) -> (Point2D, f64) {

}

impl<V: Copy> Tree2D<V> {
    pub fn new(mut points: Vec<(Point2D, V)>) -> Tree2D<V> {
        Tree2D {
            root: make_subtree(0, points.as_mut_slice()),
        }
    }

    fn find_closest_inner(&self, depth: u32, idx: usize)

    pub fn find_closest(&self) -> Option<Point2D> {
        self.root.map(|x| find_closest(0, &*x).0)
    }
}
