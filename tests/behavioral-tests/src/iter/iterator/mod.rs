mod count;
mod fold;
mod last;
mod size_hint;
mod try_fold;

use risl::gpu;

#[gpu]
pub struct SimpleIter<const N: usize> {
    pub values: [u32; N],
    pub index: usize,
}

#[gpu]
impl<const N: usize> Iterator for SimpleIter<N> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= N {
            return None;
        }

        let value = unsafe { *self.values.get_unchecked(self.index) };

        self.index += 1;

        Some(value)
    }
}
