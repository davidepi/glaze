use std::collections::HashMap;
use std::iter::FromIterator;

#[derive(Debug, Clone)]
pub struct Library<T> {
    data: Vec<(u16, String, T)>,
}

impl<T> Library<T> {
    pub fn new() -> Library<T> {
        Library { data: Vec::new() }
    }

    pub fn with_capacity(capacity: u16) -> Library<T> {
        Library {
            data: Vec::with_capacity(capacity as usize),
        }
    }

    pub fn insert(&mut self, name: &str, data: T) -> u16 {
        let id = self.data.len() as u16;
        self.data.push((id, name.to_string(), data));
        id
    }

    pub fn get(&self, id: u16) -> Option<&T> {
        if id < self.data.len() as u16 {
            unsafe { Some(&self.data.get_unchecked(id as usize).2) }
        } else {
            None
        }
    }

    pub fn len(&self) -> u16 {
        self.data.len() as u16
    }

    pub fn names(&self) -> HashMap<&str, u16> {
        self.data
            .iter()
            .map(|(id, name, _)| (name.as_str(), *id))
            .collect()
    }

    pub fn iter(&self) -> &[(u16, String, T)] {
        &self.data
    }
}

impl<T> IntoIterator for Library<T> {
    type Item = (u16, String, T);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<T> FromIterator<(u16, String, T)> for Library<T> {
    fn from_iter<I: IntoIterator<Item = (u16, String, T)>>(iter: I) -> Self {
        let data: Vec<(u16, String, T)> = iter.into_iter().collect();
        Library { data }
    }
}
