mod plot;
mod pso;
mod tsplib;

use anyhow::Result;
use tsplib::{read_tsp_file, HeuristicAlgorithm};

fn main() -> Result<()> {
    let arg = std::env::args().nth(1).unwrap_or("a280".to_string());
    let instance = format!("instances/{}.tsp", arg);
    let tsp = read_tsp_file(&instance)?;

    println!("{:?}", tsp);

    // plot::plot_algo_result(&sa, "Simulated Annealing", &plotters::style::GREEN)?;

    // println!("Best route: {:?}", sa.get_best_route().distance);
    // println!("Run time: {}ms", sa.get_run_time());

    Ok(())
}
