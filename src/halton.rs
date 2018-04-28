pub struct HaltonSequence {
    seq_num: usize,
    base: usize,
    inv_base: f64,
}

impl HaltonSequence {
    pub fn new(base: usize) -> HaltonSequence {
        HaltonSequence {
            seq_num: 1,
            base,
            inv_base: 1.0 / base as f64,
        }
    }
}

impl Iterator for HaltonSequence {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        let mut i = self.seq_num;
        let mut ret = 0.0;
        let mut r = 1.0;
        while i > 0 {
            r *= self.inv_base;
            let digit = i % self.base;
            ret += r * digit as f64;
            i /= self.base;
        }

        self.seq_num += 1;
        Some(ret)
    }
}
