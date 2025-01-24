mod aco;
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
    plot::plot_tsp_instance(tsp.clone())?;

    let mut aco = aco::AntColonyOptimization::new(&tsp, 1.0, 2.0, 0.5, 50.0, 100, 100);
    aco.solve(&tsp);

    let aco_best_route = aco.get_best_route();
    let aco_run_time = aco.get_run_time();

    println!("ACO best route: {:?}", aco_best_route.distance);
    println!("ACO run time: {} ms", aco_run_time);

    plot::plot_algo_result(&aco, "ACO", &plotters::style::BLUE)?;

    Ok(())
}
