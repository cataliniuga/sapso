use crate::tsplib::{City, HeuristicAlgorithm, Route, TspLib};
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::{collections::HashSet, time::Instant};

#[derive(Clone)]
struct Chromosome {
    route: Vec<usize>,
    distance: u64,
}

impl Chromosome {
    fn new(route: Option<Vec<usize>>, distance_matrix: &[Vec<u64>]) -> Self {
        let route = route.unwrap_or_else(|| Self::random_route(distance_matrix.len()));
        let distance = Self::calculate_distance(&route, distance_matrix);
        Chromosome { route, distance }
    }

    fn random_route(size: usize) -> Vec<usize> {
        let mut route: Vec<usize> = (0..size).collect();
        let mut rng = thread_rng();
        route.shuffle(&mut rng);
        route
    }

    fn calculate_distance(route: &[usize], distance_matrix: &[Vec<u64>]) -> u64 {
        let mut total = distance_matrix[route[route.len() - 1]][route[0]];
        for i in 1..route.len() {
            total += distance_matrix[route[i - 1]][route[i]];
        }
        total
    }

    fn nearest_neighbor_route(distance_matrix: &[Vec<u64>]) -> Vec<usize> {
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

    fn apply_2opt(&mut self, distance_matrix: &[Vec<u64>]) -> bool {
        let mut improved = false;
        let n = self.route.len();

        for i in 0..n - 2 {
            for j in i + 2..n {
                let current_distance = distance_matrix[self.route[i]][self.route[i + 1]]
                    + distance_matrix[self.route[j]][self.route[(j + 1) % n]];
                let new_distance = distance_matrix[self.route[i]][self.route[j]]
                    + distance_matrix[self.route[i + 1]][self.route[(j + 1) % n]];

                if new_distance < current_distance {
                    self.route[i + 1..=j].reverse();
                    self.distance = Self::calculate_distance(&self.route, distance_matrix);
                    improved = true;
                }
            }
        }
        improved
    }

    fn mutate(&mut self, mutation_probability: f64, distance_matrix: &[Vec<u64>]) {
        let mut rng = thread_rng();

        // Apply 2-opt with probability
        if rng.gen::<f64>() < mutation_probability {
            self.apply_2opt(distance_matrix);
        }

        // Apply random swap with probability
        if rng.gen::<f64>() < mutation_probability {
            let len = self.route.len();
            let i = rng.gen_range(0..len);
            let j = rng.gen_range(0..len);
            self.route.swap(i, j);
            self.distance = Self::calculate_distance(&self.route, distance_matrix);
        }
    }
}

pub struct GeneticAlgorithm {
    history: Vec<Route>,
    best_route: Route,
    run_time: u64,

    population_size: usize,
    number_of_generations: usize,
    mutation_probability: f64,
    elite_size: usize,
}

impl GeneticAlgorithm {
    pub fn new(
        tsp: &TspLib,
        population_size: usize,
        number_of_generations: usize,
        mutation_probability: f64,
        elite_size: usize,
    ) -> Self {
        GeneticAlgorithm {
            history: Vec::new(),
            best_route: Route::new(&tsp.cities.clone()),
            run_time: 0,
            population_size,
            number_of_generations,
            mutation_probability,
            elite_size,
        }
    }

    fn selection(&self, population: &[Chromosome]) -> Chromosome {
        let mut rng = thread_rng();
        let tournament_size = 5;
        let mut best = &population[rng.gen_range(0..population.len())];

        for _ in 1..tournament_size {
            let candidate = &population[rng.gen_range(0..population.len())];
            if candidate.distance < best.distance {
                best = candidate;
            }
        }

        best.clone()
    }
}

impl HeuristicAlgorithm for GeneticAlgorithm {
    fn solve(&mut self, tsp: &TspLib) {
        let start_time = Instant::now();

        // Initialize population with a mix of random and nearest neighbor solutions
        let mut population = Vec::with_capacity(self.population_size);

        // Add one nearest neighbor solution
        population.push(Chromosome::new(
            Some(Chromosome::nearest_neighbor_route(&tsp.distance_matrix)),
            &tsp.distance_matrix,
        ));

        // Fill rest with random solutions
        while population.len() < self.population_size {
            population.push(Chromosome::new(None, &tsp.distance_matrix));
        }

        for generation in 0..self.number_of_generations {
            population.sort_by_key(|c| c.distance);

            if generation % 100 == 0 {
                println!(
                    "Generation: {}, Best distance: {}",
                    generation, population[0].distance
                );
            }

            // Store elite solutions
            let elite = population[0..self.elite_size].to_vec();

            let mut next_population = Vec::new();
            next_population.extend(elite.clone());

            // Create new population
            while next_population.len() < self.population_size {
                let parent1 = self.selection(&population);
                let parent2 = self.selection(&population);
                let mut offspring = parent1.crossover(&parent2, &tsp.distance_matrix);
                offspring.mutate(self.mutation_probability, &tsp.distance_matrix);
                next_population.push(offspring);
            }

            // Update best solution
            population = next_population;
            population.sort_by_key(|c| c.distance);

            let best_route = Route::new(
                &population[0]
                    .route
                    .iter()
                    .map(|&city| tsp.cities[city])
                    .collect::<Vec<City>>(),
            );

            self.history.push(best_route.clone());
            if best_route.distance < self.best_route.distance {
                self.best_route = best_route;
            }
        }

        self.run_time = start_time.elapsed().as_millis() as u64;
    }

    fn get_history(&self) -> Vec<Route> {
        self.history.clone()
    }

    fn get_best_route(&self) -> Route {
        self.best_route.clone()
    }

    fn get_run_time(&self) -> u64 {
        self.run_time
    }
}
