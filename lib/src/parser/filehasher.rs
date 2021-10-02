use std::hash::Hasher;
use std::io::{Error, Read, Write};

pub struct FileHasher<T: Hasher> {
    hasher: T,
}

impl<T: Hasher> FileHasher<T> {
    pub fn new(hasher: T) -> FileHasher<T> {
        FileHasher { hasher: hasher }
    }

    pub fn hash<R: Read>(mut self, file: &mut R) -> Result<u64, Error> {
        std::io::copy(file, &mut self)?;
        Ok(self.hasher.finish())
    }
}

impl<T: Hasher> Write for FileHasher<T> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Error> {
        self.hasher.write(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::fs::File;
    use std::path::PathBuf;
    use twox_hash::XxHash64;

    use super::FileHasher;

    #[test]
    fn hash_file() -> Result<(), Box<dyn Error>> {
        let expected = 0x3330CE04D92F39D7;
        let hasher = XxHash64::with_seed(0);
        let filepath = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("resources")
            .join("checker.jpg");
        let mut file = File::open(filepath)?;
        let actual = FileHasher::new(hasher).hash(&mut file)?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
