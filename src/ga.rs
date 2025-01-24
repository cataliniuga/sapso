use std::{collections::HashSet, time::Instant};

use rand::{thread_rng, Rng};

use crate::tsplib::{City, HeuristicAlgorithm, Route, TspLib};

#[derive(Clone)]
struct Chromosome {
    route: Vec<usize>,
    distance: u64,
}

impl Chromosome {
    fn new(route: Option<Vec<usize>>, distance_matrix: &[Vec<u64>]) -> Self {
        let route = match route {
            Some(r) => r,
            None => initialize_nearest_neighbor(distance_matrix),
        };
        let distance = calculate_distance(&route, distance_matrix);

        Chromosome { route, distance }
    }

    fn crossover(&self, other: &Chromosome, distance_matrix: &[Vec<u64>]) -> Chromosome {
        let ln = self.route.len();
        let mut rng = thread_rng();
        let (left, right) = {
            let i1 = rng.gen_range(0..ln);
            let mut i2 = rng.gen_range(0..ln);
            while i2 == i1 {
                i2 = rng.gen_range(0..ln);
            }
            if i1 < i2 {
                (i1, i2)
            } else {
                (i2, i1)
            }
        };

        let mut offspring_route = vec![None; ln];
        (left..right).for_each(|i| {
            offspring_route[i] = Some(self.route[i]);
        });

        let used_cities = self.route[left..right]
            .iter()
            .cloned()
            .collect::<HashSet<usize>>();
        let mut remaining_cities = Vec::new();
        remaining_cities.extend(
            other.route[right..]
                .iter()
                .filter(|&city| !used_cities.contains(city)),
        );
        remaining_cities.extend(
            other.route[..right]
                .iter()
                .filter(|&city| !used_cities.contains(city)),
        );

        let empty_positions = (right..ln).chain(0..left);
        for (position, &city) in empty_positions.zip(remaining_cities.iter()) {
            offspring_route[position] = Some(city);
        }

        let final_route = offspring_route.into_iter().map(|x| x.unwrap()).collect();

        Chromosome::new(Some(final_route), distance_matrix)
    }

    fn mutate(&mut self, mutation_probability: f64, distance_matrix: &[Vec<u64>]) {
        let mut rng = thread_rng();

        if rng.gen::<f64>() < mutation_probability {
            let len = self.route.len();
            let i = rng.gen_range(0..len);
            let window = (len as f64 * 0.1) as usize;
            let j = (i + rng.gen_range(2..window)) % len;

            // Get start and end indices in correct order
            let (start, end) = if i < j { (i, j) } else { (j, i) };

            // 2-opt style swap
            self.route[start..=end].reverse();

            let new_distance = calculate_distance(&self.route, distance_matrix);
            if new_distance > self.distance && rng.gen::<f64>() > 0.1 {
                self.route[start..=end].reverse();
            } else {
                self.distance = new_distance;
            }
        }
    }
}

fn initialize_nearest_neighbor(distance_matrix: &[Vec<u64>]) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut current_city = rng.gen_range(0..distance_matrix.len());
    let mut unvisited = (0..distance_matrix.len())
        .filter(|&x| x != current_city)
        .collect::<Vec<usize>>();
    let mut route = vec![current_city];

    while !unvisited.is_empty() {
        let next_city = unvisited
            .iter()
            .min_by(|&&a, &&b| {
                let dist_a = distance_matrix[current_city][a];
                let dist_b = distance_matrix[current_city][b];
                dist_a.cmp(&dist_b)
            })
            .unwrap();
        let next_index = unvisited.iter().position(|&x| x == *next_city).unwrap();
        current_city = unvisited.remove(next_index);
        route.push(current_city);
    }

    route
}

fn calculate_distance(route: &[usize], distance_matrix: &[Vec<u64>]) -> u64 {
    route
        .iter()
        .zip(route.iter().skip(1))
        .map(|(a, b)| distance_matrix[*a][*b])
        .sum::<u64>()
        + distance_matrix[route[route.len() - 1]][route[0]]
}

fn selection(population: &Vec<Chromosome>) -> Chromosome {
    let total_distance = population
        .iter()
        .map(|c| (c.distance as f64).powi(-2))
        .sum::<f64>();
    let selection_point = rand::random::<f64>() * total_distance;
    let mut cumulative_distance = 0.0;

    let mut selected_chromosome = Chromosome {
        route: Vec::new(),
        distance: 0,
    };

    for chromosome in population {
        cumulative_distance += (chromosome.distance as f64).powi(-1);
        if cumulative_distance >= selection_point {
            selected_chromosome = chromosome.clone();
            break;
        }
    }

    selected_chromosome
}

pub struct GeneticAlgorithm {
    history: Vec<Route>,
    best_route: Route,
    run_time: u64,

    population_size: usize,
    number_of_generations: usize,
}

impl GeneticAlgorithm {
    pub fn new(tsp: &TspLib, population_size: usize, number_of_generations: usize) -> Self {
        GeneticAlgorithm {
            history: Vec::new(),
            best_route: Route::new(&tsp.cities.clone()),
            run_time: 0,
            population_size,
            number_of_generations,
        }
    }
}

impl HeuristicAlgorithm for GeneticAlgorithm {
    fn solve(&mut self, tsp: &crate::tsplib::TspLib) {
        let start_time = Instant::now();
        let elite_size = 2; // Keep top 2 solutions

        let mut population = (0..self.population_size)
            .map(|_| Chromosome::new(None, &tsp.distance_matrix))
            .collect::<Vec<Chromosome>>();
        for generation in 0..self.number_of_generations {
            population.sort_by(|a, b| a.distance.cmp(&b.distance));
            if generation % 100 == 0 {
                println!(
                    "Generation: {}, Best route: {}",
                    generation, population[0].distance
                );
            }

            // Store elite solutions
            let elite = population[0..elite_size].to_vec();

            let mut next_population = Vec::new();
            next_population.extend(elite.clone());

            while next_population.len() < self.population_size {
                let parent1 = selection(&population);
                let parent2 = selection(&population);
                let mut offspring1 = parent1.crossover(&parent2, &tsp.distance_matrix);
                let mut offspring2 = parent2.crossover(&parent1, &tsp.distance_matrix);
                offspring1.mutate(0.01, &tsp.distance_matrix);
                offspring2.mutate(0.01, &tsp.distance_matrix);
                next_population.push(offspring1);
                next_population.push(offspring2);
            }

            // Trim to population size if needed
            next_population.truncate(self.population_size);
            self.history.push(Route::new(
                &population[0]
                    .route
                    .iter()
                    .map(|&city| tsp.cities[city])
                    .collect::<Vec<City>>(),
            ));
            population = next_population;
        }

        let best_chromosome = population.iter().min_by_key(|c| c.distance).unwrap();
        self.best_route = Route::new(
            &best_chromosome
                .route
                .iter()
                .map(|&city| tsp.cities[city])
                .collect::<Vec<City>>(),
        );
        self.run_time = start_time.elapsed().as_millis() as u64;
    }

    fn get_history(&self) -> Vec<crate::tsplib::Route> {
        self.history.clone()
    }

    fn get_best_route(&self) -> crate::tsplib::Route {
        self.best_route.clone()
    }

    fn get_run_time(&self) -> u64 {
        self.run_time
    }
}
