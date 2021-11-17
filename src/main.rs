use automl;
use structopt::StructOpt;

/// Simulates team problem-solving using the Cognitively-Inspired Simulated Annealing Teams (CISAT) framework.
#[derive(StructOpt, Debug)]
#[structopt(author, name = "AutoML")]
struct Cli {
    /// Makes AutoML very, very chatty
    #[structopt(short, long)]
    verbose: bool,

    /// Define the type of task to perform (regression or classification)
    #[structopt(short, long)]
    task: String,

    #[structopt(short, long)]
    filepath: String,

    #[structopt(short, long)]
    header: bool,

    #[structopt(short, long)]
    index: usize,
}

fn main() {
    // Parse args
    let args = Cli::from_args();

    // Match for temperature schedule
    match args.task.to_lowercase().as_str() {
        "classification" => {
            let settings = automl::classification::Settings::default();
            let classifier = automl::classification::Classifier::default();
        }
        "regression" => {
            let settings = automl::regression::Settings::default();
            let mut regressor = automl::regression::Regressor::default();
            regressor.with_settings(settings);
            regressor.with_data_from_csv(
                args.filepath.to_lowercase().as_str(),
                args.index,
                args.header,
            );
        }
        &_ => panic!("That type of task is not supported"),
    };
}
