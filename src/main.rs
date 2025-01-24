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

    // Tuned ACO parameters:
    // - Increased iterations for better convergence
    // - Increased number of ants for better exploration
    // - Adjusted alpha/beta ratio to favor distance information
    // - Higher q value for stronger pheromone impact
    // - Higher local search probability
    let mut aco = aco::AntColonyOptimization::new(
        &tsp, 1.0,   // keep alpha
        3.0,   // increase beta more
        0.05,  // even slower decay
        300.0, // increase pheromone impact
        200,   // more ants
        250,   // more iterations
        0.4,   // more local search
    );
    aco.solve(&tsp);
    let aco_best_route = aco.get_best_route();
    let aco_run_time = aco.get_run_time();
    println!("ACO best route: {:?}", aco_best_route.distance);
    println!("ACO run time: {} ms", aco_run_time);
    plot::plot_algo_result(&aco, "ACO", &plotters::style::BLUE)?;

    // Tuned SA parameters:
    // - Higher initial temperature for better exploration
    // - Lower final temperature for better exploitation
    // - Slower cooling rate
    // - More moves per temperature
    // - Higher 2-opt probability
    let mut sa = sa::SimulatedAnnealing::new(
        &tsp,
        2000.0,                  // even higher initial temp
        0.0001,                  // lower final temp
        0.003,                   // even slower cooling
        Some(tsp.dimension * 8), // double moves again
        0.9,                     // more 2-opt
    );
    sa.solve(&tsp);
    let sa_best_route = sa.get_best_route();
    let sa_run_time = sa.get_run_time();
    println!("SA best route: {:?}", sa_best_route.distance);
    println!("SA run time: {} ms", sa_run_time);
    plot::plot_algo_result(&sa, "SA", &plotters::style::RED)?;

    // Tuned GA parameters:
    // - Larger population for better diversity
    // - Fewer generations since it plateaus early
    // - Higher mutation rate
    // - Larger elite size
    let mut ga = ga::GeneticAlgorithm::new(
        &tsp, 2000, // population_size - doubled
        3000, // number_of_generations - reduced
        0.2,  // mutation_probability - doubled
        4,    // elite_size - doubled
    );
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
