use crate::graphics::device::Device;
use crate::graphics::error::{ErrorCategory, GraphicError};

pub struct DeviceMetal {
    inner: metal::Device,
}

impl DeviceMetal {
    /// Creates an Apple Metal device.
    ///
    /// If the device_id is provided, the
    /// [registryID](https://developer.apple.com/documentation/metal/mtldevice/2915737-registryid)
    /// of the device is used look up the GPU on the system. If no GPU is found, the method will
    /// fall back to the default one.
    pub fn new(device_id: Option<u64>) -> Result<DeviceMetal, GraphicError> {
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
        .ok_or_else(|| {
            GraphicError::new(ErrorCategory::InitFailed, "Failed to find supported GPU")
        })?;
        if !pdevice.supports_BC_texture_compression() {
            return Err(GraphicError::new(
                ErrorCategory::UnsupportedFeature,
                "GPU must support BC compression",
            ));
        }
        let ret = DeviceMetal { inner: pdevice };
        Ok(ret)
    }

    pub(super) fn logical(&self) -> &metal::Device {
        &self.inner
    }
}

impl Device for DeviceMetal {}

#[cfg(test)]
mod tests {
    use super::DeviceMetal;

    #[test]
    fn create_default_device() {
        let device = DeviceMetal::new(None);
        assert!(device.is_ok())
    }
}
