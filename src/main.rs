mod aco;
mod ga;
mod plot;
mod pso;
mod sa;
mod tsplib;

use anyhow::Result;
use plotters::style::RGBColor;
use tsplib::{read_tsp_file, HeuristicAlgorithm, TspLib};

fn run_algorithm<T>(mut algorithm: T, name: &str, tsp: &TspLib, style: &RGBColor)
where
    T: HeuristicAlgorithm,
{
    algorithm.solve(tsp);
    let best_route = algorithm.get_best_route();
    let run_time = algorithm.get_run_time();
    println!("\n{} best route: {:?}", name, best_route.distance);
    println!("{} run time: {} ms\n\n", name, run_time);
    plot::plot_algo_result(&algorithm, name, style).unwrap();
}

fn main() -> Result<()> {
    let arg = std::env::args().nth(1).unwrap_or("a280".to_string());
    let instance = format!("instances/{}.tsp", arg);
    let tsp = read_tsp_file(&instance)?;

    println!("{:?}", tsp);
    plot::plot_tsp_instance(tsp.clone())?;

    let aco = aco::AntColonyOptimization::new(&tsp, 1.0, 2.0, 0.5, 50.0, 100, 100);
    run_algorithm(aco, "Ant Colony Optimization", &tsp, &plotters::style::BLUE);

    let sa = sa::SimulatedAnnealing::new(&tsp, 10000.0, 0.001, 0.1);
    run_algorithm(sa, "Simulated Annealing", &tsp, &plotters::style::RED);

    let ga = ga::GeneticAlgorithm::new(&tsp, 400, 2000);
    run_algorithm(ga, "Genetic Algorithm", &tsp, &plotters::style::GREEN);

    let pso = pso::ParticleSwarmOptimization::new(&tsp, 250, 2000, 1.5, 1.5, 0.8);
    run_algorithm(pso, "Particle Swarm Optimization", &tsp, &plotters::style::MAGENTA);

    Ok(())
}
