use std::cmp::Ordering;

// In general, floats do not implement total ordering.
// Sometimes you can guarantee that no NaN's will occur,
// and do not care about differentiating +0 and -0,
// then this type comes to the rescue.
pub struct FloatOrd<T>(pub T);

impl<T: PartialEq> PartialEq for FloatOrd<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: PartialEq> Eq for FloatOrd<T> {}

impl<T: PartialOrd> PartialOrd for FloatOrd<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }

    fn lt(&self, other: &Self) -> bool {
        self.0 < other.0
    }
    fn le(&self, other: &Self) -> bool {
        self.0 <= other.0
    }
    fn gt(&self, other: &Self) -> bool {
        self.0 > other.0
    }
    fn ge(&self, other: &Self) -> bool {
        self.0 >= other.0
    }
}

impl<T: PartialOrd> Ord for FloatOrd<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }

    fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }

    fn min(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }
}
