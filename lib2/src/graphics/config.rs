pub struct GraphicConfig {
    /// Allow using slower shared memory if the dedicated memory is not sufficient.
    pub allow_shared_vmem: bool,
    /// Maximum memory page size as 2^n bytes.
    /// Setting this value too low may result in "Allocation too big" error, while setting this
    /// value too high may reserve too much GPU memory or even result in "Not enough memory" if the
    /// value is uncredibly high.
    pub mem_order: u8,
}
