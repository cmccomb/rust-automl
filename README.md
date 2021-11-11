# AutoML with Linfa
AutoML is _Automated Machine Learning_, referring to processes and methods to make machine learning more accesible for 
a general audience. This crate builds on top of the [linfa](https://crates.io/crates/linfa) machine learning framework, 
and provides some utilities to quickly train and compare models. For instance, running the following:
```rust
fn main() {
    let data = linfa_datasets::diabetes();
    let r = automl::regression::compare_models(&data);
    print!("{}", r);
}
```
Will output this:
