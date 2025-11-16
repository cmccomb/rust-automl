//! # A cookbook of common `AutoML` tasks
//!
//! This module collects short examples demonstrating common usage patterns.
//! The examples are embedded from the examples directory and shown in the
//! generated documentation.
//!
//! ## Basic Regression (minimal example)
//! ```rust,ignore
#![doc = include_str!("../examples/minimal_regression.rs")]
//! ```
//!
//! ## Advanced Regression (maximal example)
//! ```rust,ignore
#![doc = include_str!("../examples/maximal_regression.rs")]
//! ```
//!
//! ## Wisconsin Breast Cancer Classification
//!
//! Demonstrates loading data from `data/breast_cancer.csv`, standardizing every
//! feature, and customizing the random forest search space before running the
//! leaderboard comparison.
//! ```rust,ignore
#![doc = include_str!("../examples/breast_cancer_csv.rs")]
//! ```
//!
//! ## Diabetes Progression Regression
//!
//! Shows how to impute, standardize, and tune regression algorithms on the
//! diabetes dataset that ships with the repository.
//! ```rust,ignore
#![doc = include_str!("../examples/diabetes_regression.rs")]
//! ```
