use linfa::dataset::DatasetBase;
use linfa::traits::Fit;
use linfa::traits::Transformer;
use linfa::Float;
use linfa_clustering::{Dbscan, DbscanParams, KMeans, KMeansError};
use linfa_nn::distance::{Distance, L2Dist};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, DataMut, Ix1, Ix2};
use num_traits::FromPrimitive;

struct Settings {
    cv: u64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { cv: 10 }
    }
}

trait Cluster {}
impl<F: Float> Cluster for KMeans<F, L2Dist> {}

fn compare_models<F: Float + FromPrimitive, DA: Data<Elem = F>, T, D: Distance<F>>(
    dataset: &DatasetBase<ArrayBase<DA, Ix2>, T>,
    settings: Settings,
) -> Box<dyn Cluster> {
    let n_clusters = 3;
    Box::new(KMeans::params(n_clusters).fit(dataset).unwrap())
}
