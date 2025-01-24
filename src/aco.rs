use std::time::Instant;

use rand::Rng;

use crate::tsplib::{HeuristicAlgorithm, Route, TspLib};

pub struct AntColonyOptimization {
    history: Vec<Route>,
    best_route: Route,
    run_time: u64,

    // Parameters
    pub alpha: f64,              // pheromone importance
    pub beta: f64,               // distance importance
    pub decay: f64,              // pheromone evaporation rate
    pub q: f64,                  // pheromone deposit factor
    pub ants: usize,             // number of ants
    pub iterations: usize,       // number of iterations
    pub with_local_search: bool, // apply 2-opt local search
}

impl AntColonyOptimization {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tsp: &TspLib,
        alpha: f64,
        beta: f64,
        decay: f64,
        q: f64,
        ants: usize,
        iterations: usize,
        with_local_search: bool,
    ) -> Self {
        AntColonyOptimization {
            history: Vec::new(),
            best_route: Route::new(&tsp.cities.clone()),
            run_time: 0,

            alpha,
            beta,
            decay,
            q,
            ants,
            iterations,
            with_local_search,
        }
    }

    fn construct_solution(&self, pheromone: &[Vec<f64>], tsp: &TspLib) -> Route {
        let mut rng = rand::thread_rng();
        let n = tsp.dimension;
        let mut unvisited: Vec<usize> = (0..n).collect();
        let start = rng.gen_range(0..n);
        let mut path = vec![start];
        unvisited.remove(start);

        while !unvisited.is_empty() {
            let current = *path.last().unwrap();
            let next = self.select_next_city(current, &unvisited, pheromone, tsp);
            path.push(next);
            unvisited.retain(|&x| x != next);
        }

        let route_cities: Vec<(f64, f64)> = path.iter().map(|&idx| tsp.cities[idx]).collect();

        Route::new(&route_cities)
    }

    fn construct_solution_with_local_search(&self, pheromone: &[Vec<f64>], tsp: &TspLib) -> Route {
        let initial_solution = self.construct_solution(pheromone, tsp);
        if self.with_local_search {
            initial_solution.apply_2opt()
        } else {
            initial_solution
        }
    }

    fn select_next_city(
        &self,
        current: usize,
        unvisited: &Vec<usize>,
        pheromone: &[Vec<f64>],
        tsp: &TspLib,
    ) -> usize {
        let mut rng = rand::thread_rng();
        let mut probabilities = Vec::new();
        let mut sum = 0.0;

        for &next in unvisited {
            let tau = pheromone[current][next].powf(self.alpha);
            let eta = (1.0 / tsp.distance_matrix[current][next] as f64).powf(self.beta);
            let probability = tau * eta;
            sum += probability;
            probabilities.push((next, probability));
        }

        let random_value = rng.gen::<f64>() * sum;
        let mut cumsum = 0.0;
        for (city, prob) in probabilities {
            cumsum += prob;
            if cumsum >= random_value {
                return city;
            }
        }

        *unvisited.last().unwrap()
    }

    fn update_pheromone(&self, pheromone: &mut [Vec<f64>], solutions: &Vec<Route>, tsp: &TspLib) {
        pheromone.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|value| {
                *value *= 1.0 - self.decay;
            });
        });

        for route in solutions {
            let deposit = self.q / route.distance as f64;
            let cities: Vec<usize> = route
                .cities
                .iter()
                .map(|city| tsp.cities.iter().position(|&c| c == *city).unwrap())
                .collect();

            for i in 0..cities.len() - 1 {
                let (city1, city2) = (cities[i], cities[i + 1]);
                pheromone[city1][city2] += deposit;
                pheromone[city2][city1] += deposit;
            }

            let (last, first) = (cities[cities.len() - 1], cities[0]);
            pheromone[last][first] += deposit;
            pheromone[first][last] += deposit;
        }
    }
}

impl HeuristicAlgorithm for AntColonyOptimization {
    fn solve(&mut self, tsp: &TspLib) {
        let start_time = Instant::now();

        let mut pheromone = vec![vec![1.0; tsp.dimension]; tsp.dimension];
        self.best_route = Route::new_random(&tsp.cities);

        for iteration in 0..self.iterations {
            let mut solutions = Vec::new();

            for _ in 0..self.ants {
                let solution = if self.with_local_search && iteration % 10 == 0 {
                    self.construct_solution_with_local_search(&pheromone, tsp)
                } else {
                    self.construct_solution(&pheromone, tsp)
                };

                if solution.distance < self.best_route.distance {
                    self.best_route = solution.clone();
                }

                solutions.push(solution);
            }

            self.update_pheromone(&mut pheromone, &solutions, tsp);

            self.history.push(self.best_route.clone());

            if iteration % 10 == 0 {
                println!(
                    "Iteration: {}, Best route: {}",
                    iteration, self.best_route.distance
                );
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
