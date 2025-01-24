mod aco;
mod ga;
mod plot;
mod pso;
mod sa;
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

    let mut sa = sa::SimulatedAnnealing::new(&tsp, 10000.0, 0.001, 0.1);
    sa.solve(&tsp);
    let sa_best_route = sa.get_best_route();
    let sa_run_time = sa.get_run_time();
    println!("SA best route: {:?}", sa_best_route.distance);
    println!("SA run time: {} ms", sa_run_time);
    plot::plot_algo_result(&sa, "SA", &plotters::style::RED)?;

    let mut ga = ga::GeneticAlgorithm::new(&tsp, 400, 2000);
    ga.solve(&tsp);
    let ga_best_route = ga.get_best_route();
    let ga_run_time = ga.get_run_time();
    println!("GA best route: {:?}", ga_best_route.distance);
    println!("GA run time: {} ms", ga_run_time);
    plot::plot_algo_result(&ga, "GA", &plotters::style::GREEN)?;

    let mut pso = pso::ParticleSwarmOptimization::new(&tsp, 250, 2000, 1.5, 1.5, 0.8, 10);
    pso.solve(&tsp);
    let pso_best_route = pso.get_best_route();
    let pso_run_time = pso.get_run_time();
    println!("PSO best route: {:?}", pso_best_route.distance);
    println!("PSO run time: {} ms", pso_run_time);
    plot::plot_algo_result(&pso, "PSO", &plotters::style::YELLOW)?;

    Ok(())
}
