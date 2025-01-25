mod aco;
mod ga;
mod hyper;
mod plot;
mod pso;
mod sa;
mod tsplib;

use colorful::Colorful;
use std::{fs::File, io::Write};

use anyhow::Result;
use clap::{App, Arg};
use plotters::style::RGBColor;
use tsplib::{read_tsp_file, HeuristicAlgorithm, TspLib};

fn run_algorithm<T>(mut algorithm: T, name: &str, tsp: &TspLib, style: &RGBColor)
where
    T: HeuristicAlgorithm,
{
    algorithm.solve(tsp);
    let best_route = algorithm.get_best_route();
    let run_time = algorithm.get_run_time();
    println!(
        "\n{} Best Route: {:?}",
        name.bold().rgb(style.0, style.1, style.2),
        best_route.distance
    );
    println!(
        "{} Run Time: {}ms\n\n",
        name.bold().rgb(style.0, style.1, style.2),
        run_time
    );
    plot::plot_algo_result(&algorithm, name, style).unwrap();
}

fn main() -> Result<()> {
    let matches = App::new("TSP Solver")
        .arg(
            Arg::with_name("instance")
                .help("TSP instance name")
                .default_value("a280"),
        )
        .arg(
            Arg::with_name("hyper")
                .long("hyper")
                .help("Run hyperparameter optimization")
                .takes_value(true)
                .value_name("TRIALS"),
        )
        .get_matches();

    let instance_name = matches.value_of("instance").unwrap();
    let instance = format!("instances/{}.tsp", instance_name);
    let tsp = read_tsp_file(&instance)?;

    println!("{:?}", tsp);
    plot::plot_tsp_instance(tsp.clone())?;

    if let Some(trials) = matches.value_of("hyper") {
        let num_trials = trials.parse().unwrap();
        println!(
            "Running hyperparameter optimization with {} trials...",
            num_trials
        );

        let results = hyper::optimize_hyperparameters(&tsp, num_trials);

        let mut file = File::create("hyper_results.txt")?;
        for result in &results {
            file.write_all(format!("{:?}\n", result).as_bytes())?;
        }

        let mut current_algo = String::new();
        for result in &results {
            if result.algorithm != current_algo {
                current_algo = result.algorithm.clone();
                println!("\nBest parameters for {}:", current_algo);
                println!("Distance: {}", result.distance);
                println!("Runtime: {}ms", result.runtime_ms);
                println!("Parameters: {}", result.parameters);
            }
        }

        return Ok(());
    }

    let aco = aco::AntColonyOptimization::new(&tsp, 1.0, 2.0, 0.5, 50.0, 100, 100);
    run_algorithm(aco, "Ant Colony Optimization", &tsp, &plotters::style::BLUE);

    let sa = sa::SimulatedAnnealing::new(&tsp, 1000.0, 0.001, 0.1);
    run_algorithm(sa, "Simulated Annealing", &tsp, &plotters::style::RED);

    let ga = ga::GeneticAlgorithm::new(&tsp, 400, 2000, 0.01);
    run_algorithm(ga, "Genetic Algorithm", &tsp, &plotters::style::GREEN);

    let pso = pso::ParticleSwarmOptimization::new(&tsp, 300, 4000, 1.5, 1.5, 0.8);
    run_algorithm(
        pso,
        "Particle Swarm Optimization",
        &tsp,
        &plotters::style::MAGENTA,
    );

    Ok(())
}
