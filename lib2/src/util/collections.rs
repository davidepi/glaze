const NO_FREE_SLOTS: u32 = u32::MAX;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct VacantEntry {
    generation: u32,
    next_free: u32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct OccupiedEntry<T> {
    generation: u32,
    element: T,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum Entry<T> {
    Vacant(VacantEntry),
    Occupied(OccupiedEntry<T>),
}

impl<T> Entry<T> {
    fn generation(&self) -> u32 {
        match self {
            Entry::Vacant(e) => e.generation,
            Entry::Occupied(e) => e.generation,
        }
    }
}

/// An opaque struct representing a key used to access elements in the `KeyMap`.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    index: u32,
    generation: u32,
}

impl Key {
    #[cfg(test)]
    /// Forge a default key, only for tests.
    pub fn forge() -> Self {
        Self {
            index: 0,
            generation: u32::MAX - 1,
        }
    }
}

/// A generational index map.
///
/// A data structure that combines the characteristics of a hash map and an array,
/// providing efficient management of elements through generational indices.
///
/// The `KeyMap` is particularly useful when the element type `T` does not implement
/// the `Hash` or `Ord` traits, ensuring efficient element management without relying
/// on these traits for indexing. Insertion is amortized O(1), while deletion and lookup
/// operations are O(1). The iteration over elements is not sorted.
///
/// Upon inserting an item into the `KeyMap`, a key is returned. This key must be used
/// to retrieve the associated element. However, if the element is removed in the meantime,
/// the key becomes invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyMap<T> {
    data: Vec<Entry<T>>,
    next_free: u32,
    occupied_slots: u32,
}

impl<T> Default for KeyMap<T> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            next_free: NO_FREE_SLOTS,
            occupied_slots: 0,
        }
    }
}

impl<T> KeyMap<T> {
    /// Constructs a new, empty `KeyMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use glaze_refactor::util::collections::KeyMap;
    /// let key_map: KeyMap<i32> = KeyMap::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserts an element into the `KeyMap` and returns the associated key.
    ///
    /// # Arguments
    ///
    /// * `element` - The element to be inserted into the `KeyMap`.
    ///
    /// # Returns
    ///
    /// The key associated with the inserted element, which can be used to retrieve the element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use glaze_refactor::util::collections::KeyMap;
    /// let mut key_map = KeyMap::new();
    /// let key = key_map.insert("value");
    /// ```
    #[must_use]
    pub fn insert(&mut self, element: T) -> Key {
        if self.occupied_slots == u32::MAX - 1 {
            panic!("KeyMap reached max capacity");
        }
        let key;
        if self.next_free != NO_FREE_SLOTS {
            let index = self.next_free;
            let new_entry;
            {
                // this block forces rust to drop the old_entry handle before overwriting it
                let current_entry = &self.data[index as usize];
                match current_entry {
                    Entry::Vacant(e) => {
                        // No need to update the generation: keys are not published for Vacant
                        // entries, so there is no way to reference this cell.
                        let generation = e.generation;
                        new_entry = Entry::Occupied(OccupiedEntry {
                            generation,
                            element,
                        });
                        self.next_free = e.next_free;
                        key = Key { generation, index };
                    }
                    Entry::Occupied(_) => {
                        panic!("KeyMap internal error: expected free slot is currently occupied")
                    }
                }
            }
            self.data[index as usize] = new_entry;
        } else {
            let generation = u32::MIN;
            let entry = Entry::Occupied(OccupiedEntry {
                generation,
                element,
            });
            let index = self.data.len() as u32;
            self.data.push(entry);
            key = Key { generation, index }
        }
        self.occupied_slots += 1;
        key
    }

    /// Retrieves a reference to the element associated with the provided key in the `KeyMap`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the element to be retrieved. This key is obtained upon
    /// inserting the element.
    ///
    /// # Returns
    ///
    /// An option containing a reference to the element if the key is valid and associated with an element,
    /// or `None` if the key is invalid or not associated with any element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use glaze_refactor::util::collections::{Key, KeyMap};
    /// let mut key_map = KeyMap::new();
    /// let key = key_map.insert("value");
    /// assert_eq!(key_map.get(&key), Some(&"value"));
    /// ```
    pub fn get(&self, key: &Key) -> Option<&T> {
        if let Some(entry) = self.data.get(key.index as usize) {
            match entry {
                Entry::Vacant(_) => None,
                Entry::Occupied(e) => {
                    if e.generation == key.generation {
                        Some(&e.element)
                    } else {
                        None
                    }
                }
            }
        } else {
            None
        }
    }

    /// Checks if the provided key is associated with an element in the `KeyMap`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to check for association in the `KeyMap`.
    ///
    /// # Returns
    ///
    /// A boolean value indicating whether the key is associated with an element (`true`) or not (`false`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use glaze_refactor::util::collections::{Key, KeyMap};
    /// let mut key_map = KeyMap::new();
    /// let key = key_map.insert("value");
    /// assert_eq!(key_map.contains(&key), true);
    /// ```
    pub fn contains(&self, key: &Key) -> bool {
        self.get(key).is_some() // get is already optimized enough
    }

    /// Removes and returns the element associated with the provided key from the `KeyMap`,
    /// if it exists.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the element to be removed.
    ///
    /// # Returns
    ///
    /// The removed element if the key is valid and associated with an element, or `None` if the
    /// key is invalid or not associated with any element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use glaze_refactor::util::collections::{Key, KeyMap};
    /// let mut key_map = KeyMap::new();
    /// let key = key_map.insert("value");
    /// assert_eq!(key_map.remove(key), Some("value"));
    /// ```
    pub fn remove(&mut self, key: Key) -> Option<T> {
        if key.index < self.data.len() as u32 {
            let generation = self.data[key.index as usize].generation().wrapping_add(1);
            let mut element = Entry::Vacant(VacantEntry {
                generation,
                next_free: self.next_free,
            });
            std::mem::swap(&mut self.data[key.index as usize], &mut element);
            if element.generation() == key.generation {
                if let Entry::Occupied(e) = element {
                    self.next_free = key.index;
                    self.occupied_slots -= 1;
                    Some(e.element)
                } else {
                    std::mem::swap(&mut self.data[key.index as usize], &mut element);
                    None
                }
            } else {
                std::mem::swap(&mut self.data[key.index as usize], &mut element);
                None
            }
        } else {
            None
        }
    }

    /// Checks if the `KeyMap` is empty.
    ///
    /// Note that this does not mean the underlying structure is not consuming memory.
    ///
    /// # Returns
    ///
    /// A boolean value indicating whether the `KeyMap` is empty (`true`) or not (`false`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use glaze_refactor::util::collections::KeyMap;
    /// let key_map: KeyMap<i32> = KeyMap::new();
    /// assert_eq!(key_map.is_empty(), true);
    /// ```
    pub fn is_empty(&self) -> bool {
        self.occupied_slots == 0
    }

    /// Returns the number of elements in the `KeyMap`.
    ///
    /// Note that even if the length is 0, the underlying vector could be allocated with empty entries.
    ///
    /// # Returns
    ///
    /// The number of elements in the `KeyMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use glaze_refactor::util::collections::KeyMap;
    /// let key_map: KeyMap<i32> = KeyMap::new();
    /// assert_eq!(key_map.len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.occupied_slots as usize
    }

    /// Returns an iterator over the elements of the `KeyMap`.
    ///
    /// The iteration order is not guaranteed to be in any specific order.
    ///
    /// # Returns
    ///
    /// An iterator over the elements of the `KeyMap`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use glaze_refactor::util::collections::KeyMap;
    /// let mut key_map = KeyMap::new();
    /// key_map.insert("value1");
    /// key_map.insert("value2");
    /// ```
    pub fn iter(&self) -> keymap::Iter<T> {
        keymap::Iter::new(&self.data, self.occupied_slots)
    }

    /// Removes all elements from the `KeyMap` without reallocating or shrinking the underlying
    /// data structure.
    ///
    /// # Examples
    ///
    /// ```
    /// # use glaze_refactor::util::collections::KeyMap;
    /// let mut key_map = KeyMap::new();
    /// key_map.insert("value1");
    /// key_map.insert("value2");
    /// key_map.clear();
    /// assert_eq!(key_map.len(), 0);
    /// ```
    pub fn clear(&mut self) {
        if !self.is_empty() {
            let len = self.data.len();
            for (index, old) in self.data.iter_mut().enumerate() {
                let generation = old.generation().wrapping_add(1);
                let mut next_free = (index + 1) as u32;
                if next_free as usize == len {
                    next_free = NO_FREE_SLOTS;
                }
                let new = Entry::Vacant(VacantEntry {
                    generation,
                    next_free,
                });
                *old = new;
            }
            self.next_free = 0;
            self.occupied_slots = 0;
        }
    }
}

impl<T> IntoIterator for KeyMap<T> {
    type Item = T;
    type IntoIter = keymap::IntoIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        keymap::IntoIterator::new(self.data, self.occupied_slots)
    }
}

pub mod keymap {
    use super::{Entry, Key};
    use std::iter::FusedIterator;

    /// An owning iterator for a `KeyMap`.
    ///
    /// This iterator is created by the `into_iter` method on `KeyMap`.
    ///
    /// The iteration order is not guaranteed to be in any specific order.
    pub struct IntoIterator<T> {
        data: Vec<Entry<T>>,
        elements_no: u32,
    }

    impl<T> IntoIterator<T> {
        pub(super) fn new(data: Vec<Entry<T>>, elements_no: u32) -> Self {
            Self { data, elements_no }
        }
    }

    impl<T> Iterator for IntoIterator<T> {
        /// Keys are not valid anymore after the KeyMap is consumed
        type Item = T;

        fn next(&mut self) -> Option<Self::Item> {
            let mut element = None;
            while let Some(value) = self.data.pop() {
                match value {
                    Entry::Vacant(_) => (),
                    Entry::Occupied(e) => {
                        self.elements_no -= 1;
                        element = Some(e.element);
                        break;
                    }
                }
            }
            element
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.elements_no as usize, Some(self.elements_no as usize))
        }
    }
    impl<T> FusedIterator for IntoIterator<T> {}
    impl<T> ExactSizeIterator for IntoIterator<T> {}

    /// An iterator over the elements of a `KeyMap`.
    ///
    /// This iterator is created by the `iter` method on `KeyMap`.
    ///
    /// The iteration order is not guaranteed to be in any specific order.
    pub struct Iter<'a, T> {
        data: &'a [Entry<T>],
        elements_no: u32,
        elements_returned: u32,
        index: u32,
    }

    impl<'a, T> Iter<'a, T> {
        pub(super) fn new(data: &'a [Entry<T>], elements_no: u32) -> Self {
            Self {
                data,
                elements_no,
                elements_returned: 0,
                index: 0,
            }
        }
    }

    impl<'a, T> Iterator for Iter<'a, T> {
        type Item = (Key, &'a T);

        fn next(&mut self) -> Option<Self::Item> {
            let mut element = None;
            loop {
                let index = self.index as usize;
                if index > self.data.len() {
                    break;
                }
                match &self.data[index] {
                    Entry::Vacant(_) => self.index += 1,
                    Entry::Occupied(e) => {
                        let key = Key {
                            index: self.index,
                            generation: e.generation,
                        };
                        element = Some((key, &e.element));
                        self.index += 1;
                        self.elements_returned += 1;
                    }
                }
            }
            element
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            let remaining = (self.elements_no - self.elements_returned) as usize;
            (remaining, Some(remaining))
        }
    }

    impl<'a, T> ExactSizeIterator for Iter<'a, T> {}
    impl<'a, T> FusedIterator for Iter<'a, T> {}
}

#[cfg(test)]
mod tests {
    use crate::util::collections::{Key, KeyMap};

    #[test]
    /// An element should be inserted and retrieved only with the correct key.
    fn keymap_insert_and_get() {
        let mut key_map = KeyMap::new();
        let key = key_map.insert(0);
        let got = *key_map.get(&key).unwrap();
        // Assert that the element is retrieved with the correct key
        assert_eq!(got, 0);
        // Assert that a wrong key can not retrieve the element
        let wrong_key = Key::forge();
        let got_wrong = key_map.get(&wrong_key);
        assert!(got_wrong.is_none());
    }

    #[test]
    /// After removing an element, it should be no longer accessible, even with the original key.
    fn keymap_remove_and_invalid_get() {
        let mut key_map = KeyMap::new();
        let key = key_map.insert(0);
        assert_eq!(key_map.remove(key), Some(0));
        assert_eq!(key_map.get(&key), None);
    }

    #[test]
    /// If an element is replaced, the key of the old element should not retrieve the new one.
    fn keymap_get_different_generation() {
        let mut key_map = KeyMap::new();
        // Insert the original element
        let key = key_map.insert(0);
        // Remove it and replace it with a new one
        key_map.remove(key);
        let new_key = key_map.insert(1);
        // Assert that there is exactly one element and the old key can no longer be used
        assert_eq!(key_map.len(), 1);
        assert!(key_map.get(&key).is_none());
        // Assert that the new key instead works
        assert_eq!(*key_map.get(&new_key).unwrap(), 1);
    }

    #[test]
    /// If an element is replaced, the key of the old element should not remove the new one.
    fn keymap_remove_different_generation() {
        let mut key_map = KeyMap::new();
        // Insert the original element
        let key = key_map.insert(0);
        // Remove it and replace it with a new one
        key_map.remove(key);
        let new_key = key_map.insert(1);
        // Assert that there is exactly one element and the old key can no longer be used
        assert_eq!(key_map.len(), 1);
        assert!(key_map.remove(key).is_none());
        // Assert that the new key instead works
        assert_eq!(key_map.remove(new_key).unwrap(), 1);
    }

    #[test]
    /// Clearing the collection should invalidate all keys.
    fn keymap_insert_and_clear() {
        let mut key_map = KeyMap::new();
        // Clear empty map. Should not panic, nor break functionality.
        key_map.clear();
        let mut keys = Vec::new();
        // Fill the collection
        for i in 0..33 {
            let key = key_map.insert(i);
            keys.push(key);
        }
        // Assert that all the elements are inside
        assert_eq!(key_map.len(), 33);
        // Clear the collection and assert that all elements are removed
        key_map.clear();
        assert!(key_map.is_empty());
        // Assert that no key is available anymore
        for key in keys {
            assert!(key_map.get(&key).is_none());
        }
    }

    #[test]
    /// `len` should return the amount of occupied entries.
    fn keymap_is_empty_and_len() {
        let mut key_map = KeyMap::new();
        assert!(key_map.is_empty());
        let key = key_map.insert(0);
        assert!(!key_map.is_empty());
        assert_eq!(key_map.len(), 1);
        key_map.remove(key);
        assert!(key_map.is_empty());
        assert_eq!(key_map.len(), 0);
    }
}
