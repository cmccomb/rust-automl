/// Macros for generating `with_*_settings` builder methods.
macro_rules! with_settings_methods {
    (
        $(
            $(#[$meta:meta])*
            $with_fn:ident, $field:ident, $ty:ty
        );* $(;)?
    ) => {
        $(
            $(#[$meta])*
            #[must_use]
            pub const fn $with_fn(mut self, settings: $ty) -> Self {
                self.$field = Some(settings);
                self
            }
        )*
    };
}

pub(crate) use with_settings_methods;
