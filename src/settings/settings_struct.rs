//! Settings for the automl crate

use comfy_table::{
    modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL, Attribute, Cell, Table,
};

use super::{
    Algorithm, CategoricalNBParameters, DecisionTreeClassifierParameters,
    DecisionTreeRegressorParameters, ElasticNetParameters, FinalModel, GaussianNBParameters,
    KNNClassifierParameters, KNNRegressorParameters, LassoParameters, LinearRegressionParameters,
    LinearRegressionSolverName, LogisticRegressionParameters, Metric, PreProcessing,
    RandomForestClassifierParameters, RandomForestRegressorParameters, RidgeRegressionParameters,
    RidgeRegressionSolverName, SVCParameters, SVRParameters,
};

use crate::utils::{
    debug_option, print_knn_search_algorithm, print_knn_weight_function, print_option,
};

use smartcore::{
    metrics::{accuracy, mean_absolute_error, mean_squared_error, r2},
    model_selection::KFold,
    tree::decision_tree_classifier::SplitCriterion,
};

use std::fmt::{Display, Formatter};
use std::io::{Read, Write};

/// Settings for supervised models
///
/// Any algorithms in the `skiplist` member will be skipped during training.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Settings {
    /// The metric to sort by
    pub(crate) sort_by: Metric,
    /// The type of model to train
    model_type: ModelType,
    /// The algorithms to skip
    pub(crate) skiplist: Vec<Algorithm>,
    /// The number of folds for cross-validation
    number_of_folds: usize,
    /// Whether or not to shuffle the data
    pub(crate) shuffle: bool,
    /// Whether or not to be verbose
    verbose: bool,
    /// The approach to use for the final model
    pub(crate) final_model_approach: FinalModel,
    /// The kind of preprocessing to perform
    pub(crate) preprocessing: PreProcessing,
    /// Optional settings for linear regression
    pub(crate) linear_settings: Option<LinearRegressionParameters>,
    /// Optional settings for support vector regressor
    pub(crate) svr_settings: Option<SVRParameters>,
    /// Optional settings for lasso regression
    pub(crate) lasso_settings: Option<LassoParameters<f32>>,
    /// Optional settings for ridge regression
    pub(crate) ridge_settings: Option<RidgeRegressionParameters<f32>>,
    /// Optional settings for elastic net
    pub(crate) elastic_net_settings: Option<ElasticNetParameters<f32>>,
    /// Optional settings for decision tree regressor
    pub(crate) decision_tree_regressor_settings: Option<DecisionTreeRegressorParameters>,
    /// Optional settings for random forest regressor
    pub(crate) random_forest_regressor_settings: Option<RandomForestRegressorParameters>,
    /// Optional settings for KNN regressor
    pub(crate) knn_regressor_settings: Option<KNNRegressorParameters>,
    /// Optional settings for logistic regression
    pub(crate) logistic_settings: Option<LogisticRegressionParameters<f32>>,
    /// Optional settings for random forest
    pub(crate) random_forest_classifier_settings: Option<RandomForestClassifierParameters>,
    /// Optional settings for KNN classifier
    pub(crate) knn_classifier_settings: Option<KNNClassifierParameters>,
    /// Optional settings for support vector classifier
    pub(crate) svc_settings: Option<SVCParameters>,
    /// Optional settings for decision tree classifier
    pub(crate) decision_tree_classifier_settings: Option<DecisionTreeClassifierParameters>,
    /// Optional settings for Gaussian Naive Bayes
    pub(crate) gaussian_nb_settings: Option<GaussianNBParameters<f32>>,
    /// Optional settings for Categorical Naive Bayes
    pub(crate) categorical_nb_settings: Option<CategoricalNBParameters<f32>>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            sort_by: Metric::RSquared,
            model_type: ModelType::None,
            final_model_approach: FinalModel::Best,
            skiplist: vec![
                Algorithm::LogisticRegression,
                Algorithm::RandomForestClassifier,
                Algorithm::KNNClassifier,
                Algorithm::SVC,
                Algorithm::DecisionTreeClassifier,
                Algorithm::CategoricalNaiveBayes,
                Algorithm::GaussianNaiveBayes,
                Algorithm::Linear,
                Algorithm::Lasso,
                Algorithm::Ridge,
                Algorithm::ElasticNet,
                Algorithm::SVR,
                Algorithm::DecisionTreeRegressor,
                Algorithm::RandomForestRegressor,
                Algorithm::KNNRegressor,
            ],
            preprocessing: PreProcessing::None,
            number_of_folds: 10,
            shuffle: false,
            verbose: false,
            linear_settings: None,
            svr_settings: None,
            lasso_settings: None,
            ridge_settings: None,
            elastic_net_settings: None,
            decision_tree_regressor_settings: None,
            random_forest_regressor_settings: None,
            knn_regressor_settings: None,
            logistic_settings: None,
            random_forest_classifier_settings: None,
            knn_classifier_settings: None,
            svc_settings: None,
            decision_tree_classifier_settings: None,
            gaussian_nb_settings: None,
            categorical_nb_settings: None,
        }
    }
}

impl Settings {
    /// Get the k-fold cross-validator
    pub(crate) fn get_kfolds(&self) -> KFold {
        KFold::default()
            .with_n_splits(self.number_of_folds)
            .with_shuffle(self.shuffle)
    }

    /// Get the metric to sort by
    pub(crate) fn get_metric(&self) -> fn(&Vec<f32>, &Vec<f32>) -> f32 {
        match self.sort_by {
            Metric::RSquared => r2,
            Metric::MeanAbsoluteError => mean_absolute_error,
            Metric::MeanSquaredError => mean_squared_error,
            Metric::Accuracy => accuracy,
            Metric::None => panic!("A metric must be set."),
        }
    }

    /// Creates default settings for regression
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default_regression();
    /// ```
    pub fn default_regression() -> Self {
        Settings {
            sort_by: Metric::RSquared,
            model_type: ModelType::Regression,
            final_model_approach: FinalModel::Best,
            skiplist: vec![
                Algorithm::LogisticRegression,
                Algorithm::RandomForestClassifier,
                Algorithm::KNNClassifier,
                Algorithm::SVC,
                Algorithm::DecisionTreeClassifier,
                Algorithm::CategoricalNaiveBayes,
                Algorithm::GaussianNaiveBayes,
            ],
            preprocessing: PreProcessing::None,
            number_of_folds: 10,
            shuffle: false,
            verbose: false,
            linear_settings: Some(LinearRegressionParameters::default()),
            svr_settings: Some(SVRParameters::default()),
            lasso_settings: Some(LassoParameters::default()),
            ridge_settings: Some(RidgeRegressionParameters::default()),
            elastic_net_settings: Some(ElasticNetParameters::default()),
            decision_tree_regressor_settings: Some(DecisionTreeRegressorParameters::default()),
            random_forest_regressor_settings: Some(RandomForestRegressorParameters::default()),
            knn_regressor_settings: Some(KNNRegressorParameters::default()),
            logistic_settings: None,
            random_forest_classifier_settings: None,
            knn_classifier_settings: None,
            svc_settings: None,
            decision_tree_classifier_settings: None,
            gaussian_nb_settings: None,
            categorical_nb_settings: None,
        }
    }

    /// Creates default settings for classification
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default_classification();
    /// ```
    pub fn default_classification() -> Self {
        Settings {
            sort_by: Metric::Accuracy,
            model_type: ModelType::Classification,
            final_model_approach: FinalModel::Best,
            skiplist: vec![
                Algorithm::Linear,
                Algorithm::Lasso,
                Algorithm::Ridge,
                Algorithm::ElasticNet,
                Algorithm::SVR,
                Algorithm::DecisionTreeRegressor,
                Algorithm::RandomForestRegressor,
                Algorithm::KNNRegressor,
            ],
            preprocessing: PreProcessing::None,
            number_of_folds: 10,
            shuffle: false,
            verbose: false,
            linear_settings: None,
            svr_settings: None,
            lasso_settings: None,
            ridge_settings: None,
            elastic_net_settings: None,
            decision_tree_regressor_settings: None,
            random_forest_regressor_settings: None,
            knn_regressor_settings: None,
            logistic_settings: Some(LogisticRegressionParameters::default()),
            random_forest_classifier_settings: Some(RandomForestClassifierParameters::default()),
            knn_classifier_settings: Some(KNNClassifierParameters::default()),
            svc_settings: Some(SVCParameters::default()),
            decision_tree_classifier_settings: Some(DecisionTreeClassifierParameters::default()),
            gaussian_nb_settings: Some(GaussianNBParameters::default()),
            categorical_nb_settings: Some(CategoricalNBParameters::default()),
        }
    }

    /// Load settings from a settings file
    /// ```
    /// # use automl::Settings;
    /// # let settings = Settings::default();
    /// # settings.save("tests/load_those_settings.yaml");
    /// let settings = Settings::new_from_file("tests/load_those_settings.yaml");
    /// # std::fs::remove_file("tests/load_those_settings.yaml");
    /// ```
    pub fn new_from_file(file_name: &str) -> Self {
        let mut buf: Vec<u8> = Vec::new();
        std::fs::File::open(file_name)
            .and_then(|mut f| f.read_to_end(&mut buf))
            .expect("Cannot read settings file.");
        serde_yaml::from_slice(&buf).expect("Cannot deserialize settings file.")
    }

    /// Save the current settings to a file for later use
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default_regression();
    /// settings.save("tests/save_those_settings.yaml");
    /// # std::fs::remove_file("tests/save_those_settings.yaml");
    /// ```
    pub fn save(&self, file_name: &str) {
        let serial = serde_yaml::to_string(&self).expect("Cannot serialize settings.");
        std::fs::File::create(file_name)
            .and_then(|mut f| f.write_all(serial.as_ref()))
            .expect("Cannot write settings to file.")
    }

    /// Specify number of folds for cross-validation
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default().with_number_of_folds(3);
    /// ```
    pub fn with_number_of_folds(mut self, n: usize) -> Self {
        self.number_of_folds = n;
        self
    }

    /// Specify whether or not data should be shuffled
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default().shuffle_data(true);
    /// ```
    pub fn shuffle_data(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Specify whether or not to be verbose
    /// ```
    /// # use automl::Settings;
    /// let settings = Settings::default().verbose(true);
    /// ```
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Specify what type of preprocessing should be performed
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::PreProcessing;
    /// let settings = Settings::default().with_preprocessing(PreProcessing::AddInteractions);
    /// ```
    pub fn with_preprocessing(mut self, pre: PreProcessing) -> Self {
        self.preprocessing = pre;
        self
    }

    /// Specify what type of final model to use
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::FinalModel;
    /// let settings = Settings::default().with_final_model(FinalModel::Best);
    /// ```
    pub fn with_final_model(mut self, approach: FinalModel) -> Self {
        self.final_model_approach = approach;
        self
    }

    /// Specify algorithms that shouldn't be included in comparison
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::Algorithm;
    /// let settings = Settings::default().skip(Algorithm::RandomForestRegressor);
    /// ```
    pub fn skip(mut self, skip: Algorithm) -> Self {
        self.skiplist.push(skip);
        self
    }

    /// Specify ony one algorithm to train
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::Algorithm;
    /// let settings = Settings::default().only(Algorithm::RandomForestRegressor);
    /// ```
    pub fn only(mut self, only: Algorithm) -> Self {
        self.skiplist = Self::default().skiplist;
        self.skiplist.retain(|&algo| algo != only);
        self
    }

    /// Adds a specific sorting function to the settings
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::Metric;
    /// let settings = Settings::default().sorted_by(Metric::RSquared);
    /// ```
    pub fn sorted_by(mut self, sort_by: Metric) -> Self {
        self.sort_by = sort_by;
        self
    }

    /// Specify settings for random_forest
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::RandomForestClassifierParameters;
    /// let settings = Settings::default()
    ///     .with_random_forest_classifier_settings(RandomForestClassifierParameters::default()
    ///         .with_m(100)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///         .with_n_trees(100)
    ///         .with_min_samples_split(20)
    ///     );
    /// ```
    pub fn with_random_forest_classifier_settings(
        mut self,
        settings: RandomForestClassifierParameters,
    ) -> Self {
        self.random_forest_classifier_settings = Some(settings);
        self
    }

    /// Specify settings for logistic regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::LogisticRegressionParameters;
    /// let settings = Settings::default()
    ///     .with_logistic_settings(LogisticRegressionParameters::default());
    /// ```
    pub fn with_logistic_settings(mut self, settings: LogisticRegressionParameters<f32>) -> Self {
        self.logistic_settings = Some(settings);
        self
    }

    /// Specify settings for support vector classifier
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{SVCParameters, Kernel};
    /// let settings = Settings::default()    
    ///     .with_svc_settings(SVCParameters::default()
    ///         .with_epoch(10)
    ///         .with_tol(1e-10)
    ///         .with_c(1.0)
    ///         .with_kernel(Kernel::Linear)
    ///     );
    /// ```
    pub fn with_svc_settings(mut self, settings: SVCParameters) -> Self {
        self.svc_settings = Some(settings);
        self
    }

    /// Specify settings for decision tree classifier
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::DecisionTreeClassifierParameters;
    /// let settings = Settings::default()
    ///     .with_decision_tree_classifier_settings(DecisionTreeClassifierParameters::default()
    ///         .with_min_samples_split(20)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///     );
    /// ```
    pub fn with_decision_tree_classifier_settings(
        mut self,
        settings: DecisionTreeClassifierParameters,
    ) -> Self {
        self.decision_tree_classifier_settings = Some(settings);
        self
    }

    /// Specify settings for logistic regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{KNNClassifierParameters,
    ///     KNNAlgorithmName, KNNWeightFunction, Distance};
    /// let settings = Settings::default()
    ///     .with_knn_classifier_settings(KNNClassifierParameters::default()
    ///         .with_algorithm(KNNAlgorithmName::CoverTree)
    ///         .with_k(3)
    ///         .with_distance(Distance::Euclidean)
    ///         .with_weight(KNNWeightFunction::Uniform)
    ///     );
    /// ```
    pub fn with_knn_classifier_settings(mut self, settings: KNNClassifierParameters) -> Self {
        self.knn_classifier_settings = Some(settings);
        self
    }

    /// Specify settings for Gaussian Naive Bayes
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::GaussianNBParameters;
    /// let settings = Settings::default()
    ///     .with_gaussian_nb_settings(GaussianNBParameters::default()
    ///         .with_priors(vec![1.0, 1.0])
    ///     );
    /// ```
    pub fn with_gaussian_nb_settings(mut self, settings: GaussianNBParameters<f32>) -> Self {
        self.gaussian_nb_settings = Some(settings);
        self
    }

    /// Specify settings for Categorical Naive Bayes
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::CategoricalNBParameters;
    /// let settings = Settings::default()
    ///     .with_categorical_nb_settings(CategoricalNBParameters::default()
    ///         .with_alpha(1.0)
    ///     );
    /// ```
    pub fn with_categorical_nb_settings(mut self, settings: CategoricalNBParameters<f32>) -> Self {
        self.categorical_nb_settings = Some(settings);
        self
    }

    /// Specify settings for linear regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{LinearRegressionParameters, LinearRegressionSolverName};
    /// let settings = Settings::default()
    ///     .with_linear_settings(LinearRegressionParameters::default()
    ///         .with_solver(LinearRegressionSolverName::QR)
    ///     );
    /// ```
    pub fn with_linear_settings(mut self, settings: LinearRegressionParameters) -> Self {
        self.linear_settings = Some(settings);
        self
    }

    /// Specify settings for lasso regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::LassoParameters;
    /// let settings = Settings::default()
    ///     .with_lasso_settings(LassoParameters::default()
    ///         .with_alpha(10.0)
    ///         .with_tol(1e-10)
    ///         .with_normalize(true)
    ///         .with_max_iter(10_000)
    ///     );
    /// ```
    pub fn with_lasso_settings(mut self, settings: LassoParameters<f32>) -> Self {
        self.lasso_settings = Some(settings);
        self
    }

    /// Specify settings for ridge regression
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{RidgeRegressionParameters, RidgeRegressionSolverName};
    /// let settings = Settings::default()
    ///     .with_ridge_settings(RidgeRegressionParameters::default()
    ///         .with_alpha(10.0)
    ///         .with_normalize(true)
    ///         .with_solver(RidgeRegressionSolverName::Cholesky)
    ///     );
    /// ```
    pub fn with_ridge_settings(mut self, settings: RidgeRegressionParameters<f32>) -> Self {
        self.ridge_settings = Some(settings);
        self
    }

    /// Specify settings for elastic net
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::ElasticNetParameters;
    /// let settings = Settings::default()
    ///     .with_elastic_net_settings(ElasticNetParameters::default()
    ///         .with_tol(1e-10)
    ///         .with_normalize(true)
    ///         .with_alpha(1.0)
    ///         .with_max_iter(10_000)
    ///         .with_l1_ratio(0.5)    
    ///     );
    /// ```
    pub fn with_elastic_net_settings(mut self, settings: ElasticNetParameters<f32>) -> Self {
        self.elastic_net_settings = Some(settings);
        self
    }

    /// Specify settings for KNN regressor
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{KNNRegressorParameters,
    ///     KNNAlgorithmName, KNNWeightFunction, Distance};
    /// let settings = Settings::default()
    ///     .with_knn_regressor_settings(KNNRegressorParameters::default()
    ///         .with_algorithm(KNNAlgorithmName::CoverTree)
    ///         .with_k(3)
    ///         .with_distance(Distance::Euclidean)
    ///         .with_weight(KNNWeightFunction::Uniform)
    ///     );
    /// ```
    pub fn with_knn_regressor_settings(mut self, settings: KNNRegressorParameters) -> Self {
        self.knn_regressor_settings = Some(settings);
        self
    }

    /// Specify settings for support vector regressor
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::{SVRParameters, Kernel};
    /// let settings = Settings::default()    
    ///     .with_svr_settings(SVRParameters::default()
    ///         .with_eps(1e-10)
    ///         .with_tol(1e-10)
    ///         .with_c(1.0)
    ///         .with_kernel(Kernel::Linear)
    ///     );
    /// ```
    pub fn with_svr_settings(mut self, settings: SVRParameters) -> Self {
        self.svr_settings = Some(settings);
        self
    }

    /// Specify settings for random forest
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::RandomForestRegressorParameters;
    /// let settings = Settings::default()
    ///     .with_random_forest_regressor_settings(RandomForestRegressorParameters::default()
    ///         .with_m(100)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///         .with_n_trees(100)
    ///         .with_min_samples_split(20)
    ///     );
    /// ```
    pub fn with_random_forest_regressor_settings(
        mut self,
        settings: RandomForestRegressorParameters,
    ) -> Self {
        self.random_forest_regressor_settings = Some(settings);
        self
    }

    /// Specify settings for decision tree
    /// ```
    /// # use automl::Settings;
    /// use automl::settings::DecisionTreeRegressorParameters;
    /// let settings = Settings::default()
    ///     .with_decision_tree_regressor_settings(DecisionTreeRegressorParameters::default()
    ///         .with_min_samples_split(20)
    ///         .with_max_depth(5)
    ///         .with_min_samples_leaf(20)
    ///     );
    /// ```
    pub fn with_decision_tree_regressor_settings(
        mut self,
        settings: DecisionTreeRegressorParameters,
    ) -> Self {
        self.decision_tree_regressor_settings = Some(settings);
        self
    }
}

impl Display for Settings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Prep new table
        let mut table = Table::new();

        // Get list of algorithms to skip
        let mut skiplist = String::new();
        if self.skiplist.is_empty() {
            skiplist.push_str("None ");
        } else {
            for algorithm_to_skip in &self.skiplist {
                skiplist.push_str(&format!("{}\n", algorithm_to_skip));
            }
        }

        // Build out the table
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_SOLID_INNER_BORDERS)
            .set_header(vec![
                Cell::new("Settings").add_attribute(Attribute::Bold),
                Cell::new("Value").add_attribute(Attribute::Bold),
            ])
            .add_row(vec![Cell::new("General").add_attribute(Attribute::Italic)])
            .add_row(vec!["    Model Type", &*format!("{}", self.model_type)])
            .add_row(vec!["    Verbose", &*format!("{}", self.verbose)])
            .add_row(vec!["    Sorting Metric", &*format!("{}", self.sort_by)])
            .add_row(vec!["    Shuffle Data", &*format!("{}", self.shuffle)])
            .add_row(vec![
                "    Number of CV Folds",
                &*format!("{}", self.number_of_folds),
            ])
            .add_row(vec![
                "    Pre-Processing",
                &*format!("{}", self.preprocessing),
            ])
            .add_row(vec![
                "    Skipped Algorithms",
                &skiplist[0..skiplist.len() - 1],
            ]);
        if !self.skiplist.contains(&Algorithm::Linear) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::Linear).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Solver",
                    match self.linear_settings.as_ref().unwrap().solver {
                        LinearRegressionSolverName::QR => "QR",
                        LinearRegressionSolverName::SVD => "SVD",
                    },
                ]);
        }
        if !self.skiplist.contains(&Algorithm::Ridge) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::Ridge).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Solver",
                    match self.ridge_settings.as_ref().unwrap().solver {
                        RidgeRegressionSolverName::Cholesky => "Cholesky",
                        RidgeRegressionSolverName::SVD => "SVD",
                    },
                ])
                .add_row(vec![
                    "    Alpha",
                    &*format!("{}", self.ridge_settings.as_ref().unwrap().alpha),
                ])
                .add_row(vec![
                    "    Normalize",
                    &*format!("{}", self.ridge_settings.as_ref().unwrap().normalize),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::Lasso) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::Lasso).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Alpha",
                    &*format!("{}", self.lasso_settings.as_ref().unwrap().alpha),
                ])
                .add_row(vec![
                    "    Normalize",
                    &*format!("{}", self.lasso_settings.as_ref().unwrap().normalize),
                ])
                .add_row(vec![
                    "    Maximum Iterations",
                    &*format!("{}", self.lasso_settings.as_ref().unwrap().max_iter),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.lasso_settings.as_ref().unwrap().tol),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::ElasticNet) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::ElasticNet).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Alpha",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().alpha),
                ])
                .add_row(vec![
                    "    Normalize",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().normalize),
                ])
                .add_row(vec![
                    "    Maximum Iterations",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().max_iter),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().tol),
                ])
                .add_row(vec![
                    "    L1 Ratio",
                    &*format!("{}", self.elastic_net_settings.as_ref().unwrap().l1_ratio),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::DecisionTreeRegressor) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::DecisionTreeRegressor).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(
                        self.decision_tree_regressor_settings
                            .as_ref()
                            .unwrap()
                            .max_depth,
                    ),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!(
                        "{}",
                        self.decision_tree_regressor_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_leaf
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.decision_tree_regressor_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_split
                    ),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::RandomForestRegressor) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::RandomForestRegressor).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(
                        self.random_forest_regressor_settings
                            .as_ref()
                            .unwrap()
                            .max_depth,
                    ),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!(
                        "{}",
                        self.random_forest_regressor_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_leaf
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.random_forest_regressor_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_split
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.random_forest_regressor_settings
                            .as_ref()
                            .unwrap()
                            .n_trees
                    ),
                ])
                .add_row(vec![
                    "    Number of split candidates",
                    &*print_option(self.random_forest_regressor_settings.as_ref().unwrap().m),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::KNNRegressor) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::KNNRegressor).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Number of neighbors",
                    &*format!("{}", self.knn_regressor_settings.as_ref().unwrap().k),
                ])
                .add_row(vec![
                    "    Search algorithm",
                    &print_knn_search_algorithm(
                        &self.knn_regressor_settings.as_ref().unwrap().algorithm,
                    ),
                ])
                .add_row(vec![
                    "    Weighting function",
                    &print_knn_weight_function(
                        &self.knn_regressor_settings.as_ref().unwrap().weight,
                    ),
                ])
                .add_row(vec![
                    "    Distance function",
                    &*format!(
                        "{}",
                        &self.knn_regressor_settings.as_ref().unwrap().distance
                    ),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::SVR) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::SVR).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Regularization parameter",
                    &*format!("{}", self.svr_settings.as_ref().unwrap().c),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.svr_settings.as_ref().unwrap().tol),
                ])
                .add_row(vec![
                    "    Epsilon",
                    &*format!("{}", self.svr_settings.as_ref().unwrap().eps),
                ])
                .add_row(vec![
                    "    Kernel",
                    &*format!("{}", self.svr_settings.as_ref().unwrap().kernel),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::LogisticRegression) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::LogisticRegression).add_attribute(Attribute::Italic)
                ])
                .add_row(vec!["    N/A", "N/A"]);
        }

        if !self.skiplist.contains(&Algorithm::RandomForestClassifier) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::RandomForestClassifier).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Split Criterion",
                    match self
                        .random_forest_classifier_settings
                        .as_ref()
                        .unwrap()
                        .criterion
                    {
                        SplitCriterion::Gini => "Gini",
                        SplitCriterion::Entropy => "Entropy",
                        SplitCriterion::ClassificationError => "Classification Error",
                    },
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(
                        self.random_forest_classifier_settings
                            .as_ref()
                            .unwrap()
                            .max_depth,
                    ),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!(
                        "{}",
                        self.random_forest_classifier_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_leaf
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.random_forest_classifier_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_split
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.random_forest_classifier_settings
                            .as_ref()
                            .unwrap()
                            .n_trees
                    ),
                ])
                .add_row(vec![
                    "    Number of split candidates",
                    &*print_option(self.random_forest_classifier_settings.as_ref().unwrap().m),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::KNNClassifier) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::KNNClassifier).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Number of neighbors",
                    &*format!("{}", self.knn_classifier_settings.as_ref().unwrap().k),
                ])
                .add_row(vec![
                    "    Search algorithm",
                    &print_knn_search_algorithm(
                        &self.knn_classifier_settings.as_ref().unwrap().algorithm,
                    ),
                ])
                .add_row(vec![
                    "    Weighting function",
                    &print_knn_weight_function(
                        &self.knn_classifier_settings.as_ref().unwrap().weight,
                    ),
                ])
                .add_row(vec![
                    "    Distance function",
                    &*format!(
                        "{}",
                        &self.knn_classifier_settings.as_ref().unwrap().distance
                    ),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::SVC) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::SVC).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Regularization parameter",
                    &*format!("{}", self.svc_settings.as_ref().unwrap().c),
                ])
                .add_row(vec![
                    "    Tolerance",
                    &*format!("{}", self.svc_settings.as_ref().unwrap().tol),
                ])
                .add_row(vec![
                    "    Epoch",
                    &*format!("{}", self.svc_settings.as_ref().unwrap().epoch),
                ])
                .add_row(vec![
                    "    Kernel",
                    &*format!("{}", self.svc_settings.as_ref().unwrap().kernel),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::DecisionTreeClassifier) {
            table
                .add_row(vec![
                    "    Split Criterion",
                    match self
                        .random_forest_classifier_settings
                        .as_ref()
                        .unwrap()
                        .criterion
                    {
                        SplitCriterion::Gini => "Gini",
                        SplitCriterion::Entropy => "Entropy",
                        SplitCriterion::ClassificationError => "Classification Error",
                    },
                ])
                .add_row(vec![
                    Cell::new(Algorithm::DecisionTreeClassifier).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Max Depth",
                    &*print_option(
                        self.decision_tree_classifier_settings
                            .as_ref()
                            .unwrap()
                            .max_depth,
                    ),
                ])
                .add_row(vec![
                    "    Min samples for leaf",
                    &*format!(
                        "{}",
                        self.decision_tree_classifier_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_leaf
                    ),
                ])
                .add_row(vec![
                    "    Min samples for split",
                    &*format!(
                        "{}",
                        self.decision_tree_classifier_settings
                            .as_ref()
                            .unwrap()
                            .min_samples_split
                    ),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::CategoricalNaiveBayes) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::CategoricalNaiveBayes).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Smoothing parameter",
                    &*format!("{}", self.categorical_nb_settings.as_ref().unwrap().alpha),
                ]);
        }

        if !self.skiplist.contains(&Algorithm::GaussianNaiveBayes) {
            table
                .add_row(vec![
                    Cell::new(Algorithm::GaussianNaiveBayes).add_attribute(Attribute::Italic)
                ])
                .add_row(vec![
                    "    Priors",
                    &*debug_option(self.gaussian_nb_settings.as_ref().unwrap().clone().priors),
                ]);
        }

        writeln!(f, "{table}")
    }
}

/// Model type to train
#[derive(serde::Serialize, serde::Deserialize)]
enum ModelType {
    /// No model type specified
    None,
    /// Regression model
    Regression,
    /// Classification model
    Classification,
}

impl Display for ModelType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::None => write!(f, "None"),
            ModelType::Regression => write!(f, "Regression"),
            ModelType::Classification => write!(f, "Classification"),
        }
    }
}
