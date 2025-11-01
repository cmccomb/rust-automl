use automl::settings::{
    BernoulliNBParameters, CategoricalNBParameters, ClassificationSettings,
    ClusteringAlgorithmName, ClusteringSettings, DecisionTreeClassifierParameters,
    DecisionTreeRegressorParameters, Distance, ElasticNetParameters, ExtraTreesRegressorParameters,
    FinalAlgorithm, GaussianNBParameters, KNNAlgorithmName, KNNParameters, KNNWeightFunction,
    Kernel, LassoParameters, LinearRegressionParameters, LinearRegressionSolverName,
    LogisticRegressionParameters, Metric, MultinomialNBParameters, Objective, PreProcessing,
    RandomForestClassifierParameters, RandomForestRegressorParameters, RegressionSettings,
    RidgeRegressionParameters, RidgeRegressionSolverName, SVCParameters, SVRParameters,
    XGRegressorParameters,
};
use serde_json::to_string_pretty;
use smartcore::linalg::basic::matrix::DenseMatrix;

type RegressionConfig = RegressionSettings<f64, f64, DenseMatrix<f64>, Vec<f64>>;

fn build_regression_settings() -> RegressionConfig {
    RegressionSettings::default()
        .with_number_of_folds(5)
        .shuffle_data(true)
        .verbose(true)
        .sorted_by(Metric::RSquared)
        .with_preprocessing(PreProcessing::AddInteractions)
        .with_linear_settings(
            LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::QR),
        )
        .with_lasso_settings(
            LassoParameters::default()
                .with_alpha(5.0)
                .with_tol(1e-8)
                .with_normalize(true)
                .with_max_iter(5_000),
        )
        .with_ridge_settings(
            RidgeRegressionParameters::default()
                .with_alpha(2.5)
                .with_normalize(true)
                .with_solver(RidgeRegressionSolverName::Cholesky),
        )
        .with_elastic_net_settings(
            ElasticNetParameters::default()
                .with_tol(1e-8)
                .with_normalize(true)
                .with_alpha(0.8)
                .with_max_iter(8_000)
                .with_l1_ratio(0.3),
        )
        .with_knn_regressor_settings(
            KNNParameters::default()
                .with_algorithm(KNNAlgorithmName::CoverTree)
                .with_k(7)
                .with_distance(Distance::Euclidean)
                .with_weight(KNNWeightFunction::Distance),
        )
        .with_svr_settings(
            SVRParameters::default()
                .with_eps(0.05)
                .with_tol(1e-4)
                .with_c(1.2)
                .with_kernel(Kernel::RBF(0.75)),
        )
        .with_random_forest_regressor_settings(
            RandomForestRegressorParameters::default()
                .with_m(25)
                .with_max_depth(8)
                .with_min_samples_leaf(10)
                .with_n_trees(150)
                .with_min_samples_split(15),
        )
        .with_extra_trees_settings(
            ExtraTreesRegressorParameters::default()
                .with_m(20)
                .with_max_depth(9)
                .with_min_samples_leaf(8)
                .with_min_samples_split(12)
                .with_n_trees(125)
                .with_keep_samples(true)
                .with_seed(7),
        )
        .with_decision_tree_regressor_settings(
            DecisionTreeRegressorParameters::default()
                .with_min_samples_split(12)
                .with_max_depth(7)
                .with_min_samples_leaf(6),
        )
        .with_xgboost_settings(
            XGRegressorParameters::default()
                .with_n_estimators(250)
                .with_learning_rate(0.05)
                .with_max_depth(5)
                .with_min_child_weight(2)
                .with_lambda(0.7)
                .with_gamma(0.1)
                .with_base_score(0.4)
                .with_subsample(0.85)
                .with_seed(99)
                .with_objective(Objective::MeanSquaredError),
        )
}

fn build_classification_settings() -> ClassificationSettings {
    ClassificationSettings::default()
        .with_number_of_folds(6)
        .shuffle_data(true)
        .verbose(true)
        .sorted_by(Metric::Accuracy)
        .with_preprocessing(PreProcessing::AddInteractions)
        .with_final_model(FinalAlgorithm::Best)
        .with_knn_classifier_settings(
            KNNParameters::default()
                .with_algorithm(KNNAlgorithmName::CoverTree)
                .with_k(5)
                .with_distance(Distance::Hamming)
                .with_weight(KNNWeightFunction::Uniform),
        )
        .with_decision_tree_classifier_settings(
            DecisionTreeClassifierParameters::default()
                .with_min_samples_split(12)
                .with_max_depth(6)
                .with_min_samples_leaf(6),
        )
        .with_random_forest_classifier_settings(
            RandomForestClassifierParameters::default()
                .with_m(20)
                .with_max_depth(7)
                .with_min_samples_leaf(6)
                .with_n_trees(200)
                .with_min_samples_split(12),
        )
        .with_logistic_regression_settings(
            LogisticRegressionParameters::default().with_alpha(0.1_f64),
        )
        .with_svc_settings(
            SVCParameters::default()
                .with_epoch(15)
                .with_tol(1e-4)
                .with_c(1.3)
                .with_kernel(Kernel::Polynomial(2.0, 0.5, 1.0)),
        )
        .with_bernoulli_nb_settings(
            BernoulliNBParameters::default()
                .with_alpha(0.2)
                .with_priors(vec![0.55, 0.45])
                .with_binarize(0.5_f64),
        )
        .with_gaussian_nb_settings(GaussianNBParameters::default().with_priors(vec![0.5, 0.5]))
        .with_categorical_nb_settings(CategoricalNBParameters::default().with_alpha(0.3))
        .with_multinomial_nb_settings(
            MultinomialNBParameters::default()
                .with_alpha(0.4)
                .with_priors(vec![0.6, 0.4]),
        )
}

fn build_clustering_settings() -> ClusteringSettings {
    ClusteringSettings::default()
        .with_k(4)
        .with_max_iter(250)
        .with_eps(0.35)
        .with_min_samples(12)
        .with_algorithm(ClusteringAlgorithmName::DBSCAN)
        .verbose(true)
}

fn print_settings<T>(label: &str, settings: &T) -> Result<(), serde_json::Error>
where
    T: serde::Serialize,
{
    println!("{} settings:\n{}", label, to_string_pretty(settings)?);
    Ok(())
}

fn main() -> Result<(), serde_json::Error> {
    let regression_settings = build_regression_settings();
    let classification_settings = build_classification_settings();
    let clustering_settings = build_clustering_settings();

    print_settings("Regression", &regression_settings)?;
    print_settings("Classification", &classification_settings)?;
    print_settings("Clustering", &clustering_settings)?;

    Ok(())
}
