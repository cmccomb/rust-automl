#[derive(PartialEq)]
pub(crate) enum Status {
    Starting,
    DataLoaded,
    ModelsCompared,
    FinalModelTrained,
}
