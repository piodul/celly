use std::cmp;
use std::cmp::{Ord, Ordering, PartialOrd};
use std::collections::BTreeSet;

use crate::common_geometry::{Coord, Point2D, Triangle2D};
use crate::float_ord::FloatOrd;

#[derive(PartialEq, Clone, Copy, Debug)]
struct RasterizationLineNode {
    pub upper_pt: Point2D,
    pub lower_pt: Point2D,
    pub triangle_id: usize,
}

impl RasterizationLineNode {
    fn get_intersection_at_y(&self, y: Coord) -> Coord {
        let progress = (y - self.upper_pt.1) / (self.lower_pt.1 - self.upper_pt.1);
        self.upper_pt.0 + progress * (self.lower_pt.0 - self.upper_pt.0)
    }
}

impl Eq for RasterizationLineNode {}

impl PartialOrd for RasterizationLineNode {
    fn partial_cmp(&self, other: &RasterizationLineNode) -> Option<Ordering> {
        let p1 = cmp::max(FloatOrd(self.upper_pt.1), FloatOrd(other.upper_pt.1)).0;
        let p2 = cmp::min(FloatOrd(self.lower_pt.1), FloatOrd(other.lower_pt.1)).0;
        let midpt = 0.5 * (p1 + p2);

        let l_x = self.get_intersection_at_y(midpt);
        let r_x = other.get_intersection_at_y(midpt);

        let choice = if l_x < r_x {
            Ordering::Less
        } else if l_x > r_x {
            Ordering::Greater
        } else {
            Ordering::Equal
        };

        Some(choice)
    }
}

impl Ord for RasterizationLineNode {
    fn cmp(&self, other: &RasterizationLineNode) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum RasterizationEventType {
    End,
    Begin,
}

#[derive(Clone, Copy, Debug)]
struct RasterizationEvent {
    pub etype: RasterizationEventType,
    pub time: Coord,
    pub node: RasterizationLineNode,
}

pub struct RasterizationEvents(Vec<RasterizationEvent>);

pub fn prepare_events(tris: &[Triangle2D]) -> RasterizationEvents {
    let mut events = Vec::new();
    for (id, tri) in tris.iter().enumerate() {
        let tri_edges = [(tri[1], tri[0]), (tri[2], tri[1]), (tri[0], tri[2])];
        for (a, b) in tri_edges.iter() {
            if a.1 < b.1 {
                let node = RasterizationLineNode {
                    upper_pt: *a,
                    lower_pt: *b,
                    triangle_id: id,
                };

                events.push(RasterizationEvent {
                    etype: RasterizationEventType::Begin,
                    time: a.1,
                    node,
                });
                events.push(RasterizationEvent {
                    etype: RasterizationEventType::End,
                    time: b.1,
                    node,
                });
            }
        }
    }

    events.sort_unstable_by_key(|e| (FloatOrd(e.time), e.etype));
    RasterizationEvents(events)
}

struct Broom<'a> {
    events: &'a Vec<RasterizationEvent>,
    current_event_id: usize,
    broom_state: BTreeSet<RasterizationLineNode>,
}

impl<'a> Broom<'a> {
    fn new(events: &'a RasterizationEvents) -> Broom<'a> {
        Broom {
            events: &events.0,
            current_event_id: 0,
            broom_state: BTreeSet::new(),
        }
    }

    fn advance_to(&mut self, position: Coord) {
        while self.current_event_id < self.events.len() {
            if self.events[self.current_event_id].time >= position {
                return;
            }

            let curr_event = &self.events[self.current_event_id];
            self.current_event_id += 1;

            match curr_event.etype {
                RasterizationEventType::Begin => self.broom_state.insert(curr_event.node),
                RasterizationEventType::End => self.broom_state.remove(&curr_event.node),
            };
        }
    }

    fn get_state(&self) -> &BTreeSet<RasterizationLineNode> {
        &self.broom_state
    }
}

pub fn rasterize<F: FnMut(u32, u32, Option<usize>)>(
    events: &RasterizationEvents,
    width: usize,
    height: usize,
    mut f: F,
) {
    let mut broom = Broom::new(events);
    for y in 0..height {
        let fy = y as Coord + 0.5;
        broom.advance_to(fy);

        let mut curr_tri_id = None;
        let mut curr_x = 0;
        for bevt in broom.get_state().iter() {
            let bound_x = bevt.get_intersection_at_y(fy);
            while curr_x < width && curr_x as Coord + 0.5 < bound_x {
                f(curr_x as u32, y as u32, curr_tri_id);
                curr_x += 1;
            }

            curr_tri_id = Some(bevt.triangle_id);
        }

        while curr_x < width {
            f(curr_x as u32, y as u32, curr_tri_id);
            curr_x += 1;
        }
    }
}
