use std::ops::Index;
use std::{fmt, slice};

use serde::{Deserialize, Serialize};
use thin_vec::ThinVec;

/// Set-like that just wraps a `ThinVec`.
///
/// Intended for very small sets only (where linear searching the entire set is cheap), e.g. RVSDG
/// output-value users.
#[derive(Clone, Serialize, Deserialize)]
pub struct ThinSet<T> {
    inner: ThinVec<T>,
}

impl<T> ThinSet<T>
where
    T: PartialEq,
{
    pub fn contains(&self, value: &T) -> bool {
        self.inner.iter().find(|u| *u == value).is_some()
    }

    pub(crate) fn insert(&mut self, value: T) {
        if let Some(i) = self.inner.iter().position(|u| u == &value) {
            self.inner[i] = value;
        } else {
            self.inner.push(value);
        }
    }

    pub(crate) fn remove(&mut self, user: &T) {
        if let Some(i) = self.inner.iter().position(|u| u == user) {
            self.inner.swap_remove(i);
        }
    }
}

impl<T> ThinSet<T> {
    pub fn new() -> Self {
        ThinSet {
            inner: ThinVec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn iter(&self) -> slice::Iter<'_, T> {
        self.inner.iter()
    }

    pub fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
        self.inner.iter_mut()
    }
}

impl<T> Index<usize> for ThinSet<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.inner.index(index)
    }
}

impl<'a, T> IntoIterator for &'a ThinSet<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T> Default for ThinSet<T> {
    fn default() -> Self {
        ThinSet {
            inner: Default::default(),
        }
    }
}

impl<T> PartialEq for ThinSet<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for element in self.iter() {
            if !other.contains(element) {
                return false;
            }
        }

        true
    }
}

impl<T> fmt::Debug for ThinSet<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

#[macro_export]
macro_rules! thin_set {
    () => {$crate::util::thin_set::ThinSet::new()};
    ($($x:expr),*) => ({
        let mut set = $crate::util::thin_set::ThinSet::new();

        $(set.insert($x);)*

        set
    });
    ($($x:expr,)*) => (thin_set![$($x),*]);
}
