use automl::settings::*;
use automl::*;

fn main() {
    // Totally customize settings
    let settings = Settings::default_regression()
        .with_number_of_folds(3)
        .shuffle_data(true)
        .verbose(true)
        .skip(Algorithm::RandomForestRegressor)
        .sorted_by(Metric::RSquared)
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
        .with_svr_settings(
            SVRParameters::default()
                .with_eps(1e-10)
                .with_tol(1e-10)
                .with_c(1.0)
                .with_kernel(Kernel::Linear),
        )
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

    // Load a dataset from smartcore and add it to the regressor along with the customized settings
    let mut model =
        SupervisedModel::new_from_dataset(smartcore::dataset::diabetes::load_dataset(), settings);

    // Run a model comparison with all models at default settings
    model.compare_models();
}
