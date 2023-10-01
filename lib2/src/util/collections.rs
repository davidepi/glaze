use std::ops::Index;

#[derive(Debug, Clone)]
pub struct IndexVec<T> {
    data: Vec<Option<T>>,
    free: Vec<usize>,
}

impl<T> Default for IndexVec<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            free: Default::default(),
        }
    }
}

impl<T> IndexVec<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::default(),
            free: Vec::default(),
        }
    }

    pub fn insert(&mut self, element: T) -> usize {
        if let Some(next) = self.free.pop() {
            self.data[next] = Some(element);
            next
        } else {
            let index = self.data.len();
            self.data.push(Some(element));
            index
        }
    }

    pub fn pop(&mut self, index: usize) -> Option<T> {
        let taken = self.data[index].take();
        if taken.is_some() {
            // this check is used to avoid adding to the free indexes of elements that were already
            // not in the Vec.
            self.free.push(index);
        }
        taken
    }

    /// Returns the amount of elements contained in the IndexVec. Note that these elements may not
    /// be contiguous.
    pub fn len(&self) -> usize {
        self.data.len() - &self.free.len()
    }
}

impl<T> Index<usize> for IndexVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.data[index].as_ref().unwrap()
    }
}
