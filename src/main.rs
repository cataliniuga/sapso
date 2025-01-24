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

    let mut aco = aco::AntColonyOptimization::new(&tsp, 1.0, 2.0, 0.5, 50.0, 100, 100, false);
    aco.solve(&tsp);

    let aco_best_route = aco.get_best_route();
    let aco_run_time = aco.get_run_time();

    println!("ACO best route: {:?}", aco_best_route.distance);
    println!("ACO run time: {} ms", aco_run_time);

    plot::plot_algo_result(&aco, "ACO", &plotters::style::BLUE)?;

    let mut aco_ls = aco::AntColonyOptimization::new(&tsp, 1.0, 2.0, 0.5, 50.0, 100, 100, true);
    aco_ls.solve(&tsp);

    let aco_ls_best_route = aco_ls.get_best_route();
    let aco_ls_run_time = aco_ls.get_run_time();

    println!("ACO-LS best route: {:?}", aco_ls_best_route.distance);
    println!("ACO-LS run time: {} ms", aco_ls_run_time);

    plot::plot_algo_result(&aco_ls, "ACO-LS", &plotters::style::RED)?;

    Ok(())
}
