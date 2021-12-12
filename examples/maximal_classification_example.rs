use automl::settings::*;
use automl::*;

fn main() {
    // Totally customize settings
    let settings = Settings::default_classification()
        .with_number_of_folds(3)
        .shuffle_data(true)
        .verbose(true)
        .skip(Algorithm::RandomForestClassifier)
        .sorted_by(Metric::Accuracy)
        .with_preprocessing(PreProcessing::ReplaceWithPCA {
            number_of_components: 5,
        })
        .with_random_forest_classifier_settings(
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
        .with_decision_tree_classifier_settings(
            DecisionTreeClassifierParameters::default()
                .with_min_samples_split(20)
                .with_max_depth(5)
                .with_min_samples_leaf(20),
        )
        .with_knn_classifier_settings(
            KNNClassifierParameters::default()
                .with_algorithm(KNNAlgorithmName::CoverTree)
                .with_k(3)
                .with_distance(Distance::Euclidean)
                .with_weight(KNNWeightFunction::Uniform),
        )
        .with_gaussian_nb_settings(GaussianNBParameters::default().with_priors(vec![1.0, 1.0]))
        .with_categorical_nb_settings(CategoricalNBParameters::default().with_alpha(1.0));

    // Load a dataset from smartcore and add it to the regressor
    let mut model = SupervisedModel::new_from_dataset(
        smartcore::dataset::breast_cancer::load_dataset(),
        settings,
    );

    // Run a model comparison with all models at default settings
    model.compare_models();

    // Print the results
    #[cfg(feature = "display")]
    println!("{}", model);
}
