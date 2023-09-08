use super::error::MetalError;
use crate::graphics::device::Device;

pub struct MetalDevice {
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
        .ok_or_else(|| MetalError::new("Failed to find supported GPU"))?;
        if !pdevice.supports_BC_texture_compression() {
            return Err(MetalError::new("GPU must support BC compression"));
        }
        let ret = MetalDevice { inner: pdevice };
        Ok(ret)
    }
}

impl Device for MetalDevice {
    type GraphicError = MetalError;

    fn supports_raytracing(&self) -> bool {
        self.inner.supports_raytracing()
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
