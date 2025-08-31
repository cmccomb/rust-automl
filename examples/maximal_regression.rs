use automl::settings::*;
use automl::*;

fn main() {
    let x = smartcore::linalg::basic::matrix::DenseMatrix::from_2d_array(&[
        &[234.289, 235.6, 159.0, 107.608, 1947., 60.323],
        &[259.426, 232.5, 145.6, 108.632, 1948., 61.122],
        &[258.054, 368.2, 161.6, 109.773, 1949., 60.171],
        &[284.599, 335.1, 165.0, 110.929, 1950., 61.187],
        &[328.975, 209.9, 309.9, 112.075, 1951., 63.221],
        &[346.999, 193.2, 359.4, 113.270, 1952., 63.639],
        &[365.385, 187.0, 354.7, 115.094, 1953., 64.989],
        &[363.112, 357.8, 335.0, 116.219, 1954., 63.761],
        &[397.469, 290.4, 304.8, 117.388, 1955., 66.019],
        &[419.180, 282.2, 285.7, 118.734, 1956., 67.857],
        &[442.769, 293.6, 279.8, 120.445, 1957., 68.169],
        &[444.546, 468.1, 263.7, 121.950, 1958., 66.513],
        &[482.704, 381.3, 255.2, 123.366, 1959., 68.655],
        &[502.601, 393.1, 251.4, 125.368, 1960., 69.564],
        &[518.173, 480.6, 257.2, 127.852, 1961., 69.331],
        &[554.894, 400.7, 282.7, 130.081, 1962., 70.551],
    ])
    .unwrap();

    let y: Vec<f64> = vec![
        83.0, 88.5, 88.2, 89.5, 96.2, 98.1, 99.0, 100.0, 101.2, 104.6, 108.4, 110.8, 112.6, 114.2,
        115.7, 116.9,
    ];

    // Totally customize settings
    let settings = Settings::default_regression()
        .with_number_of_folds(3)
        .shuffle_data(true)
        .verbose(true)
        .with_final_model(FinalAlgorithm::Blending {
            algorithm: Algorithm::default_linear(),
            meta_training_fraction: 0.15,
            meta_testing_fraction: 0.15,
        })
        .skip(Algorithm::default_random_forest())
        .sorted_by(Metric::RSquared)
        .with_preprocessing(PreProcessing::AddInteractions)
        .with_linear_settings(
            LinearRegressionParameters::default().with_solver(LinearRegressionSolverName::QR),
        )
        .with_lasso_settings(
            LassoParameters::default()
                .with_alpha(1.0)
                .with_tol(1e-4)
                .with_normalize(true)
                .with_max_iter(1000),
        )
        .with_ridge_settings(
            RidgeRegressionParameters::default()
                .with_alpha(1.0)
                .with_normalize(true)
                .with_solver(RidgeRegressionSolverName::Cholesky),
        )
        .with_elastic_net_settings(
            ElasticNetParameters::default()
                .with_tol(1e-4)
                .with_normalize(true)
                .with_alpha(1.0)
                .with_max_iter(1000)
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
        //         .with_eps(0.1)
        //         .with_tol(1e-3)
        //         .with_c(1.0)
        //         .with_kernel(Kernel::Linear),
        // )
        .with_random_forest_regressor_settings(
            RandomForestRegressorParameters::default()
                .with_m(1)
                .with_max_depth(5)
                .with_min_samples_leaf(1)
                .with_n_trees(10)
                .with_min_samples_split(2),
        )
        .with_decision_tree_regressor_settings(
            DecisionTreeRegressorParameters::default()
                .with_min_samples_split(2)
                .with_max_depth(15)
                .with_min_samples_leaf(1),
        );

    // Load a dataset from smartcore and add it to the regressor along with the customized settings
    let mut model = SupervisedModel::build(x, y, settings);

    // Run a model comparison with all models at default settings
    model.train();

    // Print the results
    println!("{}", model);
}
