use automl::classification::settings::*;
use automl::classification::*;

fn main() {
    // Define a default regressor. This include settings, but no data yet.
    let mut classifier = Classifier::default();

    // Totally customize settings
    let settings = Settings::default()
        .with_number_of_folds(3)
        .shuffle_data(true)
        .verbose(true)
        .skip(Algorithm::RandomForest)
        .sorted_by(Metric::Accuracy)
        .with_random_forest_settings(
            RandomForestClassifierParameters::default()
                .with_m(100)
                .with_max_depth(5)
                .with_min_samples_leaf(20)
                .with_n_trees(100)
                .with_min_samples_split(20),
        )
        .with_logistic_settings(LogisticRegressionParameters::default())
        .with_svc_settings(
            SVCParameters::default()
                .with_epoch(10)
                .with_tol(1e-10)
                .with_c(1.0)
                .with_kernel(Kernel::Linear),
        )
        .with_decision_tree_settings(
            DecisionTreeClassifierParameters::default()
                .with_min_samples_split(20)
                .with_max_depth(5)
                .with_min_samples_leaf(20),
        )
        .with_knn_settings(
            KNNClassifierParameters::default()
                .with_algorithm(KNNAlgorithmName::CoverTree)
                .with_k(3)
                .with_distance(Distance::Euclidean)
                .with_weight(KNNWeightFunction::Uniform),
        )
        .with_gaussian_nb_settings(GaussianNBParameters::default().with_priors(vec![1.0, 1.0]))
        .with_categorical_nb_settings(CategoricalNBParameters::default().with_alpha(1.0));
    classifier.with_settings(settings);

    // Load a dataset from smartcore and add it to the regressor
    classifier.with_dataset(smartcore::dataset::diabetes::load_dataset());

    // Run a model comparison with all models at default settings
    classifier.compare_models();
}
