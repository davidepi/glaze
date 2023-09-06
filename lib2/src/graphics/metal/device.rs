use super::error::MetalError;

struct MetalDevice {
    inner: metal::Device,
}

impl MetalDevice {
    /// Creates an Apple Metal device.
    ///
    /// If the device_id is provided, the
    /// (registryID)[https://developer.apple.com/documentation/metal/mtldevice/2915737-registryid]
    /// of the device is used look up the GPU on the system. If no GPU is found, the method will
    /// fall back to the default one.
    pub fn new(device_id: Option<u64>) -> Result<MetalDevice, MetalError> {
        let pdevice = if let Some(id) = device_id {
            match metal::Device::all()
                .into_iter()
                .find(|device| device.registry_id() == id)
            {
                Some(d) => Some(d),
                None => {
                    log::error!(" Could not find device with id {id}. Using default device.");
                    metal::Device::system_default()
                }
            }
        } else {
            metal::Device::system_default()
        }
        .ok_or_else(|| MetalError::new("Failed to find metal device".to_string()))?;
        let ret = MetalDevice { inner: pdevice };
        Ok(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::MetalDevice;

    #[test]
    fn create_default_device() {
        let device = MetalDevice::new(None);
        assert!(device.is_ok())
    }
}
