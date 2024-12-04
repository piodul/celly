use rayon::prelude::*;
use std::cmp::{self, Ord, Ordering, PartialOrd};
use std::collections::BTreeSet;

use crate::common_geometry::{Coord, Point2D, Triangle2D};
use crate::float_ord::FloatOrd;

fn get_intersection_at_y(upper: Point2D, lower: Point2D, y: Coord) -> Coord {
    let progress = (y - upper.1) / (lower.1 - upper.1);
    upper.0 + progress * (lower.0 - upper.0)
}

// fn get_intersection_at_x(upper: Point2D, lower: Point2D, x: Coord) -> Coord {
//     // Dirty trick, I can't think too deeply right now
//     get_intersection_at_y((upper.1, upper.0), (lower.1, lower.0), x)
// }

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

    fn calculate_area_clipped_to_unit_square(&self, origin: Point2D) -> f64 {
        let mut ret = self.clone();
        ret.upper_y -= origin.1;
        ret.lower_y -= origin.1;
        if ret.lower_y <= 0.0 || ret.upper_y >= 1.0 {
            // No intersection, the trapezoid is either above or below
            // the unit square
            return 0.0;
        }

        ret.left_x[0] -= origin.0;
        ret.left_x[1] -= origin.0;
        ret.right_x[0] -= origin.0;
        ret.right_x[1] -= origin.0;

        // Shave off the parts above y = 0.0 and below y = 1.0
        if ret.upper_y < 0.0 {
            ret.left_x[0] = ret.get_left_intersection_at_y(0.0);
            ret.right_x[0] = ret.get_right_intersection_at_y(0.0);
            ret.upper_y = 0.0;
        }
        if ret.lower_y > 1.0 {
            ret.left_x[1] = ret.get_left_intersection_at_y(1.0);
            ret.right_x[1] = ret.get_right_intersection_at_y(1.0);
            ret.lower_y = 1.0;
        }

        let calculate_right_angled_trapezoid_clipped_to_unit_square =
            |mut lower_x: Coord, mut upper_x: Coord, lower_y: Coord, upper_y: Coord| -> f64 {
                // Consider a right-angled trapezoid with corners in (0, 0) and (0, 1).
                // `lower_x` and `upper_x` define the x-coordinates of the two right corners.
                // This function calculates the area of such a trapezoid.
                if lower_x > upper_x {
                    // Should give the same result, but will allow us
                    // to simplify code
                    std::mem::swap(&mut lower_x, &mut upper_x);
                }

                // Moment of intersection of the right edge with x == 0
                let m_0 = match (lower_x > 0.0, upper_x > 0.0) {
                    (true, true) => upper_y,
                    (false, false) => lower_y,
                    (false, true) => {
                        // Abuse, but whatever
                        get_intersection_at_y((lower_y, upper_x), (upper_y, lower_x), 0.0)
                    }
                    _ => unreachable!(), // because v_lower <= v_upper
                };

                // Moment of intersection of the right edge with x == 1
                let m_1 = match (lower_x > 1.0, upper_x > 1.0) {
                    (true, true) => upper_y,
                    (false, false) => lower_y,
                    (false, true) => {
                        // Abuse, but whatever
                        get_intersection_at_y((lower_y, upper_x), (upper_y, lower_x), 1.0)
                    }
                    _ => unreachable!(), // because v_lower <= v_upper
                };

                let b_0 = lower_x.clamp(0.0, 1.0);
                let b_1 = upper_x.clamp(0.0, 1.0);

                // TODO: Perhaps can be further simplified
                let area = 0.5 * (b_1 + b_0) * (m_1 - m_0) + (lower_y - m_1);

                area
            };

        let area = calculate_right_angled_trapezoid_clipped_to_unit_square(
            ret.right_x[0],
            ret.right_x[1],
            ret.lower_y,
            ret.upper_y,
        ) - calculate_right_angled_trapezoid_clipped_to_unit_square(
            ret.left_x[0],
            ret.left_x[1],
            ret.lower_y,
            ret.upper_y,
        );

        area
    }
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
            let y_end = std::cmp::min(y + self.rows_per_chunk, self.height);
            assert_eq!((y_end - y) * self.width, chunk.sample_counts.len());

            chunk.replay(y, y_end, self.width, &mut f);

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
    const MIN_PIXEL_COUNT_PER_CHUNK: usize = 16 * 4096;
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
                            let coverage = trapezoid
                                .calculate_area_clipped_to_unit_square((curr_x as Coord, fy));
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
