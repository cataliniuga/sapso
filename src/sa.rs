use crate::tsplib::*;
use rand::prelude::*;
use std::time::Instant;

pub struct SimulatedAnnealing {
    history: Vec<Route>,
    best_route: Route,
    run_time: u64,

    // Parameters
    pub initial_temperature: f64,
    pub final_temperature: f64,
    pub cooling_rate: f64,
    pub moves_per_temp: usize,
    pub two_opt_probability: f64,
}

impl SimulatedAnnealing {
    pub fn new(
        tsp: &TspLib,
        initial_temperature: f64,
        final_temperature: f64,
        cooling_rate: f64,
        moves_per_temp: Option<usize>,
        two_opt_probability: f64,
    ) -> Self {
        SimulatedAnnealing {
            history: Vec::new(),
            best_route: Route::new(&tsp.cities),
            run_time: 0,
            initial_temperature,
            final_temperature,
            cooling_rate,
            moves_per_temp: moves_per_temp.unwrap_or(tsp.dimension * 2),
            two_opt_probability,
        }
    }

    fn initialize_nearest_neighbor(&self, tsp: &TspLib) -> Route {
        let mut rng = rand::thread_rng();
        let mut current_city = rng.gen_range(0..tsp.dimension);
        let mut unvisited = (0..tsp.dimension)
            .filter(|&x| x != current_city)
            .collect::<Vec<usize>>();
        let mut route_indices = vec![current_city];

        while !unvisited.is_empty() {
            let next_city = unvisited
                .iter()
                .min_by(|&&a, &&b| {
                    let dist_a = tsp.distance_matrix[current_city][a];
                    let dist_b = tsp.distance_matrix[current_city][b];
                    dist_a.cmp(&dist_b)
                })
                .unwrap();
            let next_index = unvisited.iter().position(|&x| x == *next_city).unwrap();
            current_city = unvisited.remove(next_index);
            route_indices.push(current_city);
        }

        let route_cities = route_indices
            .iter()
            .map(|&idx| tsp.cities[idx])
            .collect::<Vec<City>>();

        Route::new(&route_cities)
    }

    fn apply_2opt_move(&self, route: &Route, i: usize, j: usize) -> Route {
        route.two_opt_move(i, j)
    }

    fn apply_swap_move(&self, route: &Route) -> Route {
        let mut rng = rand::thread_rng();
        let mut new_cities = route.cities.clone();
        let i = rng.gen_range(0..new_cities.len());
        let j = rng.gen_range(0..new_cities.len());
        new_cities.swap(i, j);
        Route::new(&new_cities)
    }

    fn generate_neighbor(&self, route: &Route, temperature: f64) -> Route {
        let mut rng = rand::thread_rng();

        // Adaptive move selection based on temperature
        let temp_ratio = temperature / self.initial_temperature;
        let two_opt_prob = self.two_opt_probability * temp_ratio;

        if rng.gen::<f64>() < two_opt_prob {
            // 2-opt move
            let n = route.cities.len();
            let i = rng.gen_range(0..n - 1);
            let j = rng.gen_range(i + 1..n);
            self.apply_2opt_move(route, i, j)
        } else {
            // Swap move
            self.apply_swap_move(route)
        }
    }

    fn calculate_acceptance_probability(
        &self,
        current_distance: u64,
        new_distance: u64,
        temperature: f64,
    ) -> f64 {
        if new_distance <= current_distance {
            1.0
        } else {
            let delta = (new_distance - current_distance) as f64;
            (-delta / temperature).exp()
        }
    }
}

impl HeuristicAlgorithm for SimulatedAnnealing {
    fn solve(&mut self, tsp: &TspLib) {
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();
        let mut temperature = self.initial_temperature;
        let mut epoch = 0;

        // Initialize with nearest neighbor
        let mut current_route = self.initialize_nearest_neighbor(tsp);
        self.best_route = current_route.clone();

        // Main loop
        while temperature > self.final_temperature {
            let mut improved = false;

            if epoch % 100 == 0 {
                println!(
                    "Epoch: {}, Temperature: {:.2}, Best Distance: {}",
                    epoch, temperature, self.best_route.distance
                );
            }

            // Moves at current temperature
            for _ in 0..self.moves_per_temp {
                let new_route = self.generate_neighbor(&current_route, temperature);

                let acceptance_probability = self.calculate_acceptance_probability(
                    current_route.distance,
                    new_route.distance,
                    temperature,
                );

                if acceptance_probability > rng.gen::<f64>() {
                    if new_route.distance < self.best_route.distance {
                        self.best_route = new_route.clone();
                        improved = true;
                    }
                    current_route = new_route;
                }
            }

            // Store history and update temperature
            self.history.push(current_route.clone());

            // Adaptive cooling rate based on improvement
            let cooling_factor = if improved {
                self.cooling_rate * 0.95 // Slower cooling when improving
            } else {
                self.cooling_rate * 1.05 // Faster cooling when stuck
            };

            temperature *= 1.0 - cooling_factor.min(0.1).max(0.001);
            epoch += 1;
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
