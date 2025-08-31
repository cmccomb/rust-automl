use automl::settings::{
    DecisionTreeRegressorParameters, Distance, ElasticNetParameters, KNNAlgorithmName,
    KNNRegressorParameters, KNNWeightFunction, LassoParameters, LinearRegressionParameters,
    LinearRegressionSolverName, Metric, PreProcessing, RandomForestRegressorParameters,
    RidgeRegressionParameters, RidgeRegressionSolverName, Settings,
};
use smartcore::linalg::basic::matrix::DenseMatrix;

fn main() {
    let regressor_settings: Settings<f64, f64, DenseMatrix<f64>, Vec<f64>> =
        automl::Settings::default_regression()
            .with_number_of_folds(3)
            .shuffle_data(true)
            .verbose(true)
            .sorted_by(Metric::RSquared)
            .with_preprocessing(PreProcessing::AddInteractions)
            .with_linear_settings(
                LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::QR),
            )
            .with_lasso_settings(
                LassoParameters::default()
                    .with_alpha(10.0)
                    .with_tol(1e-10)
                    .with_normalize(true)
                    .with_max_iter(10_000),
            )
            .with_ridge_settings(
                RidgeRegressionParameters::default()
                    .with_alpha(10.0)
                    .with_normalize(true)
                    .with_solver(RidgeRegressionSolverName::Cholesky),
            )
            .with_elastic_net_settings(
                ElasticNetParameters::default()
                    .with_tol(1e-10)
                    .with_normalize(true)
                    .with_alpha(1.0)
                    .with_max_iter(10_000)
                    .with_l1_ratio(0.5),
            )
            .with_knn_regressor_settings(
                KNNRegressorParameters::default()
                    .with_algorithm(KNNAlgorithmName::CoverTree)
                    .with_k(3)
                    .with_distance(Distance::Euclidean)
                    .with_weight(KNNWeightFunction::Uniform),
            )
            // .with_svr_settings(
            //     SVRParameters::default()
            //         .with_eps(1e-10)
            //         .with_tol(1e-10)
            //         .with_c(1.0)
            //         .with_kernel(Kernel::Linear),
            // )
            .with_random_forest_regressor_settings(
                RandomForestRegressorParameters::default()
                    .with_m(100)
                    .with_max_depth(5)
                    .with_min_samples_leaf(20)
                    .with_n_trees(100)
                    .with_min_samples_split(20),
            )
            .with_decision_tree_regressor_settings(
                DecisionTreeRegressorParameters::default()
                    .with_min_samples_split(20)
                    .with_max_depth(5)
                    .with_min_samples_leaf(20),
            );

    // let classifier_settings = automl::Settings::default_classification()
    //     .with_number_of_folds(3)
    //     .shuffle_data(true)
    //     .verbose(true)
    //     .sorted_by(Metric::Accuracy)
    //     .with_preprocessing(PreProcessing::AddInteractions)
    //     .with_random_forest_classifier_settings(
    //         RandomForestClassifierParameters::default()
    //             .with_m(100)
    //             .with_max_depth(5)
    //             .with_min_samples_leaf(20)
    //             .with_n_trees(100)
    //             .with_min_samples_split(20),
    //     )
    //     .with_logistic_settings(LogisticRegressionParameters::default())
    //     .with_svc_settings(
    //         SVCParameters::default()
    //             .with_epoch(10)
    //             .with_tol(1e-10)
    //             .with_c(1.0)
    //             .with_kernel(Kernel::Linear),
    //     )
    //     .with_decision_tree_classifier_settings(
    //         DecisionTreeClassifierParameters::default()
    //             .with_min_samples_split(20)
    //             .with_max_depth(5)
    //             .with_min_samples_leaf(20),
    //     )
    //     .with_knn_classifier_settings(
    //         KNNClassifierParameters::default()
    //             .with_algorithm(KNNAlgorithmName::CoverTree)
    //             .with_k(3)
    //             .with_distance(Distance::Hamming)
    //             .with_weight(KNNWeightFunction::Uniform),
    //     )
    //     .with_gaussian_nb_settings(GaussianNBParameters::default().with_priors(vec![1.0, 1.0]))
    //     .with_categorical_nb_settings(CategoricalNBParameters::default().with_alpha(1.0));

    println!("{regressor_settings}");
    // println!("{}", classifier_settings)
}
