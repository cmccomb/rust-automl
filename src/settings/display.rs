//! Helpers for rendering `Settings` as a configuration table.
//!
//! # Examples
//! ```
//! use automl::{DenseMatrix, Settings};
//!
//! let settings = Settings::<f64, f64, DenseMatrix<f64>, Vec<f64>>::default();
//! println!("{settings}");
//! ```

use std::fmt::{Display, Formatter, Write};

use comfy_table::{
    Attribute, Cell, Table, modifiers::UTF8_SOLID_INNER_BORDERS, presets::UTF8_FULL,
};

use smartcore::linalg::basic::arrays::Array1;
use smartcore::linalg::traits::cholesky::CholeskyDecomposable;
use smartcore::linalg::traits::qr::QRDecomposable;
use smartcore::linalg::traits::svd::SVDDecomposable;
use smartcore::numbers::floatnum::FloatNumber;
use smartcore::numbers::realnum::RealNumber;

use crate::utils::display::print_option;

use super::{Algorithm, LinearRegressionSolverName, RidgeRegressionSolverName, Settings};

impl<INPUT, OUTPUT, InputArray, OutputArray> Settings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber,
    OUTPUT: FloatNumber,
    InputArray: CholeskyDecomposable<INPUT> + SVDDecomposable<INPUT> + QRDecomposable<INPUT>,
    OutputArray: Array1<OUTPUT>,
{
    /// Create a table preloaded with headers and styling.
    fn init_table() -> Table {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_SOLID_INNER_BORDERS)
            .set_header(vec![
                Cell::new("Settings").add_attribute(Attribute::Bold),
                Cell::new("Value").add_attribute(Attribute::Bold),
            ])
            .add_row(vec![Cell::new("General").add_attribute(Attribute::Italic)]);
        table
    }

    /// Build a string representation of skipped algorithms.
    fn skiplist_string(&self) -> String {
        let mut skiplist = String::new();
        if self.skiplist.is_empty() {
            skiplist.push_str("None ");
        } else {
            for algorithm_to_skip in &self.skiplist {
                writeln!(&mut skiplist, "{algorithm_to_skip}").expect("writing to string");
            }
        }
        skiplist
    }

    /// Append general settings rows.
    fn add_general_rows(&self, table: &mut Table) {
        let skip = self.skiplist_string();
        table
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
            .add_row(vec!["    Skipped Algorithms", &skip[0..skip.len() - 1]]);
    }

    /// Append linear regression settings.
    fn add_linear_rows(&self, table: &mut Table) {
        table
            .add_row(vec![
                Cell::new(Algorithm::<INPUT, OUTPUT, InputArray, OutputArray>::default_linear())
                    .add_attribute(Attribute::Italic),
            ])
            .add_row(vec![
                "    Solver",
                match self.linear_settings.as_ref().unwrap().solver {
                    LinearRegressionSolverName::QR => "QR",
                    LinearRegressionSolverName::SVD => "SVD",
                },
            ]);
    }

    /// Append ridge regression settings.
    fn add_ridge_rows(&self, table: &mut Table) {
        table
            .add_row(vec![
                Cell::new(Algorithm::<INPUT, OUTPUT, InputArray, OutputArray>::default_ridge())
                    .add_attribute(Attribute::Italic),
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

    /// Append LASSO regression settings.
    fn add_lasso_rows(&self, table: &mut Table) {
        table
            .add_row(vec![
                Cell::new(Algorithm::<INPUT, OUTPUT, InputArray, OutputArray>::default_lasso())
                    .add_attribute(Attribute::Italic),
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

    /// Append elastic net settings.
    fn add_elastic_net_rows(&self, table: &mut Table) {
        table
            .add_row(vec![
                Cell::new(
                    Algorithm::<INPUT, OUTPUT, InputArray, OutputArray>::default_elastic_net(),
                )
                .add_attribute(Attribute::Italic),
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

    /// Append decision tree regressor settings.
    fn add_decision_tree_rows(&self, table: &mut Table) {
        table
            .add_row(vec![
                Cell::new(
                    Algorithm::<INPUT, OUTPUT, InputArray, OutputArray>::default_decision_tree(),
                )
                .add_attribute(Attribute::Italic),
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

    /// Append random forest regressor settings.
    fn add_random_forest_rows(&self, table: &mut Table) {
        table
            .add_row(vec![
                Cell::new(
                    Algorithm::<INPUT, OUTPUT, InputArray, OutputArray>::default_random_forest(),
                )
                .add_attribute(Attribute::Italic),
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
                "    Number of Trees",
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
}

impl<INPUT, OUTPUT, InputArray, OutputArray> Display
    for Settings<INPUT, OUTPUT, InputArray, OutputArray>
where
    INPUT: FloatNumber + RealNumber,
    OUTPUT: FloatNumber,
    InputArray: CholeskyDecomposable<INPUT> + SVDDecomposable<INPUT> + QRDecomposable<INPUT>,
    OutputArray: Array1<OUTPUT>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut table = Self::init_table();
        self.add_general_rows(&mut table);
        if !self.skiplist.contains(&Algorithm::default_linear()) {
            self.add_linear_rows(&mut table);
        }
        if !self.skiplist.contains(&Algorithm::default_ridge()) {
            self.add_ridge_rows(&mut table);
        }
        if !self.skiplist.contains(&Algorithm::default_lasso()) {
            self.add_lasso_rows(&mut table);
        }
        if !self.skiplist.contains(&Algorithm::default_elastic_net()) {
            self.add_elastic_net_rows(&mut table);
        }
        if !self.skiplist.contains(&Algorithm::default_decision_tree()) {
            self.add_decision_tree_rows(&mut table);
        }
        if !self.skiplist.contains(&Algorithm::default_random_forest()) {
            self.add_random_forest_rows(&mut table);
        }
        writeln!(f, "{table}")
    }
}
