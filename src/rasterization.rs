use rayon::prelude::*;
use std::cmp::{self, Ord, Ordering, PartialOrd};
use std::collections::BTreeSet;

use arrayvec::ArrayVec;

use crate::common_geometry::{det2, Coord, Point2D, Triangle2D};
use crate::float_ord::FloatOrd;

fn get_intersection_at_y(upper: Point2D, lower: Point2D, y: Coord) -> Coord {
    let progress = (y - upper.1) / (lower.1 - upper.1);
    upper.0 + progress * (lower.0 - upper.0)
}

// fn get_intersection_at_x(upper: Point2D, lower: Point2D, x: Coord) -> Coord {
//     // Dirty trick, I can't think too deeply right now
//     get_intersection_at_y((upper.1, upper.0), (lower.1, lower.0), x)
// }

fn calculate_clipped_area(trapezoid: &Trapezoid, origin: Point2D) -> f64 {
    // Normalize to the rect upper left corner
    let mut v1: ArrayVec<Point2D, 16> = ArrayVec::new();
    let mut v2: ArrayVec<Point2D, 16> = ArrayVec::new();

    v1.push((trapezoid.left_x[1] - origin.0, trapezoid.lower_y - origin.1));
    v1.push((
        trapezoid.right_x[1] - origin.0,
        trapezoid.lower_y - origin.1,
    ));
    v1.push((
        trapezoid.right_x[0] - origin.0,
        trapezoid.upper_y - origin.1,
    ));
    v1.push((trapezoid.left_x[0] - origin.0, trapezoid.upper_y - origin.1));

    let mut verts = &mut v1;
    let mut new_verts = &mut v2;

    for _ in 0..4 {
        // Clip da edges
        let mut prev_vert = match verts.last().cloned() {
            Some(v) => v,
            None => return 0.0, // Completely clipped
        };
        for curr_vert in verts.iter() {
            let prev_clipped = prev_vert.1 < 0.0;
            let curr_clipped = curr_vert.1 < 0.0;
            match (prev_clipped, curr_clipped) {
                (true, true) => {
                    // Do nothing, the whole edge is clipped
                }
                (true, false) => {
                    let x = get_intersection_at_y(*curr_vert, prev_vert, 0.0);
                    new_verts.push((x, 0.0));
                    new_verts.push(*curr_vert);
                }
                (false, true) => {
                    let x = get_intersection_at_y(*curr_vert, prev_vert, 0.0);
                    new_verts.push((x, 0.0));
                }
                _ => {
                    new_verts.push(*curr_vert);
                }
            }

            prev_vert = *curr_vert;
        }

        // Rotate by 90 degrees
        for v in new_verts.iter_mut() {
            *v = (1.0 - v.1, v.0);
        }

        std::mem::swap(&mut verts, &mut new_verts);
        new_verts.clear();
    }

    // Finally, calculate the area
    let mut area = 0.0;
    let mut prev_vert = match verts.last().cloned() {
        Some(v) => v,
        None => return 0.0,
    };
    for curr_vert in verts.iter() {
        area += det2(curr_vert.0, curr_vert.1, prev_vert.0, prev_vert.1);
        prev_vert = *curr_vert;
    }
    // println!("area: {}", area * 0.5);
    area * 0.5
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum RasterizationEventType {
    End,
    Begin,
}

#[derive(PartialEq, Clone, Debug)]
struct Trapezoid {
    upper_y: Coord,
    lower_y: Coord,
    left_x: [Coord; 2],
    right_x: [Coord; 2],
}

impl Trapezoid {
    fn get_left_intersection_at_y(&self, y: Coord) -> Coord {
        get_intersection_at_y(
            (self.left_x[0], self.upper_y),
            (self.left_x[1], self.lower_y),
            y,
        )
    }

    fn get_right_intersection_at_y(&self, y: Coord) -> Coord {
        get_intersection_at_y(
            (self.right_x[0], self.upper_y),
            (self.right_x[1], self.lower_y),
            y,
        )
    }

    fn validate(&self) {
        debug_assert!(self.upper_y <= self.lower_y);
        debug_assert!(self.left_x[0] <= self.right_x[0]);
        debug_assert!(self.left_x[1] <= self.right_x[1]);
    }

    // // Perhaps it should be more efficient than the clipping method,
    // // but it does not account for some special cases right now,
    // // so I'm using a real clipping algorithm for the time being.
    // fn calculate_area_clipped_to_unit_square(&self, origin: Point2D) -> f64 {
    //     let mut ret = self.clone();
    //     ret.upper_y -= origin.1;
    //     ret.lower_y -= origin.1;
    //     if ret.lower_y >= 1.0 || ret.upper_y <= 0.0 {
    //         // No intersection, the trapezoid is either above or below
    //         // the unit square
    //         return 0.0;
    //     }

    //     ret.left_x[0] -= origin.0;
    //     ret.left_x[1] -= origin.0;
    //     ret.right_x[0] -= origin.0;
    //     ret.right_x[1] -= origin.0;

    //     // Shave off the parts above y = 0.0 and below y = 1.0
    //     if ret.upper_y < 0.0 {
    //         ret.left_x[0] = ret.get_left_intersection_at_y(0.0);
    //         ret.right_x[0] = ret.get_right_intersection_at_y(0.0);
    //         ret.upper_y = 0.0;
    //     }
    //     if ret.lower_y > 1.0 {
    //         ret.left_x[1] = ret.get_left_intersection_at_y(1.0);
    //         ret.right_x[1] = ret.get_right_intersection_at_y(1.0);
    //         ret.lower_y = 1.0;
    //     }

    //     let compute_area_to_the_right = |x: Coord| -> f64 {
    //         println!(" compute_area_to_the_right: begin {x}");

    //         let mut a = 0.0;

    //         // Right "wing"
    //         match (ret.right_x[0] <= x, ret.right_x[1] <= x) {
    //             (true, true) => {
    //                 // Everything is to the left
    //                 println!("  right: everything to the left - return");
    //                 return a;
    //             }
    //             (true, false) => {
    //                 let midpt = get_intersection_at_x(
    //                     (ret.right_x[0], ret.upper_y),
    //                     (ret.right_x[1], ret.lower_y),
    //                     x,
    //                 );
    //                 a += 0.5 * (1.0 - midpt) * (ret.right_x[1] - x);
    //                 println!("  right: true, false, increase by {}", a);
    //                 return a;
    //             }
    //             (false, true) => {
    //                 let midpt = get_intersection_at_x(
    //                     (ret.right_x[0], ret.upper_y),
    //                     (ret.right_x[1], ret.lower_y),
    //                     x,
    //                 );
    //                 a += 0.5 * midpt * (ret.right_x[0] - x);
    //                 println!("  right: false, true, increase by {}", a);
    //                 return a;
    //             }
    //             (false, false) => {
    //                 a += 0.5 * (ret.right_x[1] - ret.right_x[0]).abs(); // * 1.0 (height)
    //                 println!("  right: false, false, increase by {}", a);
    //             }
    //         }

    //         // The rectangle part
    //         let rect_left = std::cmp::max(FloatOrd(ret.left_x[0]), FloatOrd(ret.left_x[1])).0;
    //         let rect_right = std::cmp::min(FloatOrd(ret.right_x[0]), FloatOrd(ret.right_x[1])).0;
    //         if rect_left <= x {
    //             println!("  center: rectangle intersects, area {}", rect_right - x);
    //             a += rect_right - x; // * 1.0 (height)
    //             return a;
    //         }
    //         println!(
    //             "  center: rectangle does not intersect, area {}",
    //             rect_right - rect_left
    //         );
    //         a += rect_right - rect_left;

    //         // Left "wing"
    //         match (ret.left_x[0] <= x, ret.left_x[1] <= x) {
    //             (true, true) => {
    //                 // Everything is to the left (not very likely but can happen)
    //                 println!("  left: everything to the left - return");
    //                 return a;
    //             }
    //             (true, false) => {
    //                 let midpt = get_intersection_at_x(
    //                     (ret.left_x[0], ret.upper_y),
    //                     (ret.left_x[1], ret.lower_y),
    //                     x,
    //                 );
    //                 let incr = (ret.left_x[1] - x) - 0.5 * (ret.left_x[1] - x) * (1.0 - midpt);
    //                 a += incr;
    //                 println!("  left: true, false, increase by {}", incr);
    //                 return a;
    //             }
    //             (false, true) => {
    //                 let midpt = get_intersection_at_x(
    //                     (ret.left_x[0], ret.upper_y),
    //                     (ret.left_x[1], ret.lower_y),
    //                     x,
    //                 );
    //                 let incr = (ret.left_x[0] - x) - 0.5 * (ret.left_x[0] - x) * midpt;
    //                 a += incr;
    //                 println!("  left: false, true, increase by {}", incr);
    //                 return a;
    //             }
    //             (false, false) => {
    //                 let incr = 0.5 * (ret.left_x[1] - ret.left_x[0]).abs(); // * 1.0 (height)
    //                 a += incr;
    //                 println!("  left: false, false, increase by {}", incr);
    //             }
    //         }

    //         a
    //     };

    //     let trap = &ret;

    //     println!("starting computation");
    //     let ret = compute_area_to_the_right(0.0) - compute_area_to_the_right(1.0);
    //     if ret < 0.0 || ret > 1.0 {
    //         panic!("{ret}, {:?}", trap);
    //     }
    //     ret
    // }
}

impl Eq for Trapezoid {}

impl PartialOrd for Trapezoid {
    fn partial_cmp(&self, other: &Trapezoid) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Trapezoid {
    fn cmp(&self, other: &Trapezoid) -> Ordering {
        let p1 = cmp::max(FloatOrd(self.upper_y), FloatOrd(other.upper_y)).0;
        let p2 = cmp::min(FloatOrd(self.lower_y), FloatOrd(other.lower_y)).0;
        let midpt = 0.5 * (p1 + p2);

        let l_x = self.get_left_intersection_at_y(midpt);
        let r_x = other.get_left_intersection_at_y(midpt);

        if l_x < r_x {
            Ordering::Less
        } else if l_x > r_x {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone)]
enum TrapezoidKind {
    Upper,
    Lower,
}

struct RasterizationEvent {
    etype: RasterizationEventType,
    tkind: TrapezoidKind,
    trapezoid: Trapezoid,
    triangle_id: usize,
}

impl RasterizationEvent {
    fn time(&self) -> f64 {
        match self.etype {
            RasterizationEventType::End => self.trapezoid.lower_y,
            RasterizationEventType::Begin => self.trapezoid.upper_y,
        }
    }
}

pub struct RasterizationEvents(Vec<RasterizationEvent>);

pub fn prepare_aa_events(tris: &[Triangle2D]) -> RasterizationEvents {
    let mut events = Vec::new();
    for (id, tri) in tris.iter().enumerate() {
        // We need to divide the triangle into two triangles which have one
        // side that is parallel to the X axis
        let mut tri = *tri;
        tri.sort_by_key(|p| (FloatOrd(p.1), FloatOrd(p.0)));

        let mut generate_events = |tkind: TrapezoidKind, trapezoid: &Trapezoid| {
            trapezoid.validate();

            events.push(RasterizationEvent {
                etype: RasterizationEventType::Begin,
                tkind,
                trapezoid: trapezoid.clone(),
                triangle_id: id,
            });
            events.push(RasterizationEvent {
                etype: RasterizationEventType::End,
                tkind,
                trapezoid: trapezoid.clone(),
                triangle_id: id,
            });
        };

        if tri[0].1 == tri[1].1 {
            // The upper side is already parallel
            // tri[0] has lower X than tri[1], therefore we use it
            generate_events(
                TrapezoidKind::Lower,
                &Trapezoid {
                    upper_y: tri[0].1,
                    lower_y: tri[2].1,
                    left_x: [tri[0].0, tri[2].0],
                    right_x: [tri[1].0, tri[2].0],
                },
            );
        } else if tri[1].1 == tri[2].1 {
            // The lower side is already parallel
            generate_events(
                TrapezoidKind::Upper,
                &Trapezoid {
                    upper_y: tri[0].1,
                    lower_y: tri[2].1,
                    left_x: [tri[0].0, tri[1].0],
                    right_x: [tri[0].0, tri[2].0],
                },
            );
        } else {
            let split_x = get_intersection_at_y(tri[0], tri[2], tri[1].1);

            // Is tri[1] on the left or right side of the tri[0] -> tri[2] line?
            if split_x <= tri[1].0 {
                generate_events(
                    TrapezoidKind::Upper,
                    &Trapezoid {
                        upper_y: tri[0].1,
                        lower_y: tri[1].1,
                        left_x: [tri[0].0, split_x],
                        right_x: [tri[0].0, tri[1].0],
                    },
                );
                generate_events(
                    TrapezoidKind::Lower,
                    &Trapezoid {
                        upper_y: tri[1].1,
                        lower_y: tri[2].1,
                        left_x: [split_x, tri[2].0],
                        right_x: [tri[1].0, tri[2].0],
                    },
                );
            } else {
                generate_events(
                    TrapezoidKind::Upper,
                    &Trapezoid {
                        upper_y: tri[0].1,
                        lower_y: tri[1].1,
                        left_x: [tri[0].0, tri[1].0],
                        right_x: [tri[0].0, split_x],
                    },
                );
                generate_events(
                    TrapezoidKind::Lower,
                    &Trapezoid {
                        upper_y: tri[1].1,
                        lower_y: tri[2].1,
                        left_x: [tri[1].0, tri[2].0],
                        right_x: [split_x, tri[2].0],
                    },
                );
            }
        }
    }

    events.sort_unstable_by_key(|e| (FloatOrd(e.time()), e.etype));
    RasterizationEvents(events)
}

#[derive(Clone)]
struct Broom<'a> {
    events: &'a [RasterizationEvent],
    broom_state: BTreeSet<(&'a Trapezoid, usize, TrapezoidKind)>,
}

impl<'a> Broom<'a> {
    fn new(events: &'a RasterizationEvents) -> Broom<'a> {
        Broom {
            events: &events.0,
            broom_state: BTreeSet::new(),
        }
    }

    // Returns the events that were processed.
    fn advance_to(&mut self, position: Coord) -> &'a [RasterizationEvent] {
        let old_events = self.events;
        while let Some((curr_event, tail)) = self.events.split_first() {
            if curr_event.time() >= position {
                break;
            }
            self.events = tail;

            match curr_event.etype {
                RasterizationEventType::Begin => self.broom_state.insert((
                    &curr_event.trapezoid,
                    curr_event.triangle_id,
                    curr_event.tkind,
                )),
                RasterizationEventType::End => self.broom_state.remove(&(
                    &curr_event.trapezoid,
                    curr_event.triangle_id,
                    curr_event.tkind,
                )),
            };
        }
        &old_events[..old_events.len() - self.events.len()]
    }

    fn get_state(&self) -> &BTreeSet<(&'a Trapezoid, usize, TrapezoidKind)> {
        &self.broom_state
    }
}

#[derive(Debug)]
struct ScanlineEvent<'a> {
    time: f64,
    etype: RasterizationEventType,
    tkind: TrapezoidKind,
    triangle_id: usize,
    trapezoid: &'a Trapezoid,
}

pub type CoveringTriangleInfo = (usize, f64);

pub struct CoverageChunk {
    sample_counts: Vec<usize>,
    samples: Vec<CoveringTriangleInfo>,
}

pub struct ComputedCoverage {
    width: usize,
    height: usize,
    rows_per_chunk: usize,
    chunks: Vec<CoverageChunk>,
}

impl ComputedCoverage {
    pub fn replay(&self, mut f: impl FnMut(u32, u32, &[CoveringTriangleInfo])) {
        let mut y = 0;
        for chunk in self.chunks.iter() {
            assert_eq!(self.rows_per_chunk * self.width, chunk.sample_counts.len());

            chunk.replay(
                y,
                std::cmp::min(y + self.rows_per_chunk, self.height),
                self.width,
                &mut f,
            );

            y += self.rows_per_chunk;
        }
    }

    pub fn rows_per_chunk(&self) -> usize {
        self.rows_per_chunk
    }

    pub fn chunks(&self) -> &[CoverageChunk] {
        self.chunks.as_slice()
    }
}

impl CoverageChunk {
    pub fn replay(
        &self,
        y_start: usize,
        y_end: usize,
        width: usize,
        mut f: impl FnMut(u32, u32, &[CoveringTriangleInfo]),
    ) {
        let mut y = y_start;
        let mut curr_samples = self.samples.as_slice();
        for row in 0..y_end - y_start {
            for x in 0..width {
                let scount = self.sample_counts[row * width + x];
                let (head_samples, tail_samples) = curr_samples.split_at(scount);
                f(x as u32, y as u32, head_samples);
                curr_samples = tail_samples;
            }
            y += 1;
        }

        assert_eq!(curr_samples.len(), 0);
    }
}

pub fn rasterize<'a>(
    events: &'a RasterizationEvents,
    width: usize,
    height: usize,
) -> ComputedCoverage {
    let mut broom = Broom::new(events);

    // Split the image into horizontal stripes and process in parallel
    const MIN_PIXEL_COUNT_PER_CHUNK: usize = 4096;
    let rows_per_chunk = (MIN_PIXEL_COUNT_PER_CHUNK + width - 1) / width;

    let mut y = 0;
    let work_items = std::iter::from_fn(|| {
        if y >= height {
            return None;
        }

        let start_y = y;
        y += rows_per_chunk;
        let end_y = std::cmp::min(y, height);

        broom.advance_to(start_y as Coord);
        Some((broom.clone(), start_y, end_y))
    })
    .collect::<Vec<_>>();

    let chunks = work_items
        .into_par_iter()
        .map(|(mut b, start_y, end_y)| rasterize_scanlines(&mut b, width, start_y, end_y - start_y))
        .collect();

    ComputedCoverage {
        width,
        height,
        rows_per_chunk,
        chunks,
    }
}

fn rasterize_scanlines<'a>(
    broom: &'a mut Broom,
    width: usize,
    y_offset: usize,
    height: usize,
) -> CoverageChunk {
    assert_ne!(width, 0);
    assert_ne!(height, 0);

    let mut sample_counts = Vec::with_capacity(width * height);
    let mut samples = Vec::new();

    let mut scanline_events: Vec<ScanlineEvent> = Vec::new();
    let mut coverage_info: Vec<(usize, TrapezoidKind, &'a Trapezoid)> = Vec::new();

    for y in y_offset..y_offset + height {
        let fy = y as Coord;
        scanline_events.clear();
        coverage_info.clear();

        // Update the broom
        let broom_events = broom.advance_to(fy + 1.0);

        let mut push_events =
            |tkind: TrapezoidKind, trapezoid: &'a Trapezoid, triangle_id: usize, lower_y: Coord| {
                let start_time = if trapezoid.left_x[0] < trapezoid.left_x[1] {
                    trapezoid.get_left_intersection_at_y(fy) - 1.0
                } else {
                    trapezoid.get_left_intersection_at_y(lower_y) - 1.0
                };
                let end_time = if trapezoid.right_x[0] > trapezoid.right_x[1] {
                    trapezoid.get_right_intersection_at_y(fy)
                } else {
                    trapezoid.get_right_intersection_at_y(lower_y)
                };
                scanline_events.push(ScanlineEvent {
                    time: start_time,
                    etype: RasterizationEventType::Begin,
                    tkind,
                    triangle_id,
                    trapezoid,
                });
                scanline_events.push(ScanlineEvent {
                    time: end_time,
                    etype: RasterizationEventType::End,
                    tkind,
                    triangle_id,
                    trapezoid,
                });
            };

        // Add events based on the items which left the broom (including those
        // that were inserted and immediately removed)
        for e in broom_events {
            match e.etype {
                RasterizationEventType::End => {
                    push_events(e.tkind, &e.trapezoid, e.triangle_id, e.trapezoid.lower_y);
                }
                RasterizationEventType::Begin => {} // Nothing
            }
        }

        // Add events based on the state of the broom
        for e in broom.get_state().iter() {
            push_events(e.2, &e.0, e.1, fy + 1.0);
        }

        scanline_events.sort_unstable_by_key(|e| FloatOrd(e.time));

        let mut curr_x = 0usize;
        'scanline_events: for evt in scanline_events.iter() {
            // Rasterize
            let bound_x = evt.time;
            let bound_ux = cmp::min(bound_x.ceil() as usize, width);
            let bound_ux = cmp::max(bound_ux, curr_x);
            let segment_length = bound_ux - curr_x;
            match coverage_info.as_slice() {
                [(t, _, _)] => {
                    // Very common case: only one triangle covers the pixel.
                    // The triangles that we rasterize are disjoint
                    // and cover the whole image, so we can assume that
                    // the whole pixel is covered.
                    sample_counts.extend(std::iter::repeat(1).take(segment_length));
                    samples.extend(std::iter::repeat((*t, 1.0)).take(segment_length));
                    curr_x = bound_ux;
                }
                [(t1, _, _), (t2, _, _)] if t1 == t2 => {
                    // Similar situation as above, but happens if we have
                    // two trapezoids from the same triangle.
                    sample_counts.extend(std::iter::repeat(1).take(segment_length));
                    samples.extend(std::iter::repeat((*t1, 1.0)).take(segment_length));
                    curr_x = bound_ux;
                }
                _ => {
                    // Rasterize, but update the coverage on each step
                    sample_counts
                        .extend(std::iter::repeat(coverage_info.len()).take(segment_length));
                    while curr_x < bound_ux {
                        // Update coverage
                        for (tri, _, trapezoid) in coverage_info.iter() {
                            let coverage = calculate_clipped_area(trapezoid, (curr_x as Coord, fy));
                            samples.push((*tri, coverage));
                        }

                        curr_x += 1;
                    }
                }
            }

            // Early exit if we reached the end of the line
            if curr_x >= width {
                break 'scanline_events;
            }

            // Process the next event
            match evt.etype {
                RasterizationEventType::End => {
                    let idx = coverage_info
                        .iter()
                        .enumerate()
                        .find_map(|(idx, (t, tk, _))| {
                            (*t == evt.triangle_id && *tk == evt.tkind).then_some(idx)
                        })
                        .expect("tried to remove a triangle that is not being considered");
                    coverage_info.swap_remove(idx);
                }
                RasterizationEventType::Begin => {
                    coverage_info.push((evt.triangle_id, evt.tkind, evt.trapezoid));
                }
            }
        }

        // The triangles are supposed to cover the whole area and we have events
        // both for the left and right triangle sides, so we should have
        // covered the whole line
        assert_eq!(curr_x, width);
    }

    CoverageChunk {
        sample_counts,
        samples,
    }
}

#[allow(dead_code)]
fn dump_obj(trapezoids: impl Iterator<Item = Trapezoid>, origin: Point2D) {
    println!("# OBJ start");

    let mut tcount = 0;
    for t in trapezoids {
        tcount += 1;
        println!(
            "v {} {} {}",
            t.left_x[0] - origin.0,
            0.0,
            t.upper_y - origin.1,
        );
        println!(
            "v {} {} {}",
            t.right_x[0] - origin.0,
            0.0,
            t.upper_y - origin.1,
        );
        println!(
            "v {} {} {}",
            t.right_x[1] - origin.0,
            0.0,
            t.lower_y - origin.1,
        );
        println!(
            "v {} {} {}",
            t.left_x[1] - origin.0,
            0.0,
            t.lower_y - origin.1,
        );
    }

    for i in 0..tcount {
        println!("f {} {} {} {}", 4 * i + 1, 4 * i + 2, 4 * i + 3, 4 * i + 4);
    }

    println!("# OBJ end");
}
