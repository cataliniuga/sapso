use prettytable::{row, Table};
use rand::Rng;
use rayon::prelude::*;
use serde::Serialize;
use std::sync::{Arc, Mutex};

use crate::{
    aco::AntColonyOptimization,
    ga::GeneticAlgorithm,
    pso::ParticleSwarmOptimization,
    sa::SimulatedAnnealing,
    tsplib::{HeuristicAlgorithm, TspLib},
};

#[derive(Debug, Clone, Serialize)]
pub struct OptimizationResult {
    pub algorithm: String,
    pub parameters: String,
    pub distance: u64,
    pub runtime_ms: u64,
}

#[derive(Debug)]
struct AcoParams {
    alpha: f64,        // pheromone importance [0.5..4.0]
    beta: f64,         // distance importance [1.0..5.0]
    decay: f64,        // evaporation rate [0.01..0.5]
    q: f64,            // pheromone deposit factor [1.0..500.0]
    ants: usize,       // number of ants [50..500]
    iterations: usize, // number of iterations [200..2000]
}

#[derive(Debug)]
struct SaParams {
    initial_temp: f64, // [1000.0..50000.0]
    final_temp: f64,   // [0.0001..0.1]
    cooling_rate: f64, // [0.001..0.3]
}

#[derive(Debug)]
struct GaParams {
    population_size: usize, // [100..2000]
    generations: usize,     // [100..5000]
    mutation_rate: f64,     // [0.001..0.3]
}

#[derive(Debug)]
struct PsoParams {
    num_particles: usize,  // [50..1000]
    iterations: usize,     // [200..5000]
    cognitive_weight: f64, // [0.5..4.0]
    social_weight: f64,    // [0.5..4.0]
    inertia_weight: f64,   // [0.1..0.9]
}

pub fn optimize_hyperparameters(tsp: &TspLib, num_trials: usize) -> Vec<OptimizationResult> {
    let tsp = Arc::new(tsp.clone());
    let results = Arc::new(Mutex::new(Vec::new()));

    (0..num_trials).into_par_iter().for_each(|_| {
        let mut rng = rand::thread_rng();
        let tsp = Arc::clone(&tsp);
        let results = Arc::clone(&results);

        // ACO with wider ranges
        let aco_params = AcoParams {
            alpha: rng.gen_range(1.0..5.0),
            beta: rng.gen_range(1.0..8.0),
            decay: rng.gen_range(0.02..0.6),
            q: rng.gen_range(10.0..600.0),
            ants: rng.gen_range(100..600),
            iterations: rng.gen_range(500..3000),
        };

        let mut aco = AntColonyOptimization::new(
            &tsp,
            aco_params.alpha,
            aco_params.beta,
            aco_params.decay,
            aco_params.q,
            aco_params.ants,
            aco_params.iterations,
        );

        aco.solve(&tsp);
        let aco_result = OptimizationResult {
            algorithm: "ACO".to_string(),
            parameters: format!("{:?}", aco_params),
            distance: aco.get_best_route().distance,
            runtime_ms: aco.get_run_time(),
        };

        // SA with wider ranges
        let sa_params = SaParams {
            initial_temp: rng.gen_range(5000.0..80000.0),
            final_temp: rng.gen_range(0.00001..0.2),
            cooling_rate: rng.gen_range(0.0005..0.4),
        };

        let mut sa = SimulatedAnnealing::new(
            &tsp,
            sa_params.initial_temp,
            sa_params.final_temp,
            sa_params.cooling_rate,
        );

        sa.solve(&tsp);
        let sa_result = OptimizationResult {
            algorithm: "SA".to_string(),
            parameters: format!("{:?}", sa_params),
            distance: sa.get_best_route().distance,
            runtime_ms: sa.get_run_time(),
        };

        // GA with wider ranges
        let ga_params = GaParams {
            population_size: rng.gen_range(200..3000),
            generations: rng.gen_range(500..7000),
            mutation_rate: rng.gen_range(0.001..0.4),
        };

        let mut ga = GeneticAlgorithm::new(
            &tsp,
            ga_params.population_size,
            ga_params.generations,
            ga_params.mutation_rate,
        );

        ga.solve(&tsp);
        let ga_result = OptimizationResult {
            algorithm: "GA".to_string(),
            parameters: format!("{:?}", ga_params),
            distance: ga.get_best_route().distance,
            runtime_ms: ga.get_run_time(),
        };

        let pso_params = PsoParams {
            num_particles: rng.gen_range(100..2000),
            iterations: rng.gen_range(500..7000),
            cognitive_weight: rng.gen_range(1.0..5.0),
            social_weight: rng.gen_range(1.0..5.0),
            inertia_weight: rng.gen_range(0.05..0.95),
        };

        let mut pso = ParticleSwarmOptimization::new(
            &tsp,
            pso_params.num_particles,
            pso_params.iterations,
            pso_params.cognitive_weight,
            pso_params.social_weight,
            pso_params.inertia_weight,
        );

        pso.solve(&tsp);
        let pso_result = OptimizationResult {
            algorithm: "PSO".to_string(),
            parameters: format!("{:?}", pso_params),
            distance: pso.get_best_route().distance,
            runtime_ms: pso.get_run_time(),
        };

        let mut results = results.lock().unwrap();
        results.push(aco_result);
        results.push(sa_result);
        results.push(ga_result);
        results.push(pso_result);
    });

    let results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    let mut final_results = results;
    final_results.sort_by(|a, b| {
        if a.algorithm == b.algorithm {
            a.distance.cmp(&b.distance)
        } else {
            a.algorithm.cmp(&b.algorithm)
        }
    });

    print_results_table(&final_results);

    final_results
}

fn print_results_table(results: &[OptimizationResult]) {
    let mut current_algo = String::new();
    let mut table = Table::new();

    for result in results {
        if result.algorithm != current_algo {
            if !current_algo.is_empty() {
                table.printstd();
                println!("\n");
                table = Table::new();
            }
            current_algo = result.algorithm.clone();

            table.add_row(row![bFg => format!("{} Results", current_algo)]);
            table.add_row(row![bFg => "Parameters", "Distance", "Runtime (ms)"]);
        }

        table.add_row(row![result.parameters, result.distance, result.runtime_ms]);
    }

    table.printstd();
}
