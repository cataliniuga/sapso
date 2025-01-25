use crate::tsplib::*;
use rand::prelude::*;

pub struct SimulatedAnnealing {
    history: Vec<Route>,
    best_route: Route,
    run_time: u64,

    pub temperature: f64,
    pub cooling_rate: f64,
    pub min_temperature: f64,
}

impl SimulatedAnnealing {
    pub fn new(tsp: &TspLib, temperature: f64, cooling_rate: f64, min_temperature: f64) -> Self {
        SimulatedAnnealing {
            history: Vec::new(),
            best_route: Route::new(&tsp.cities),
            run_time: 0,

            temperature,
            cooling_rate,
            min_temperature,
        }
    }
}

impl HeuristicAlgorithm for SimulatedAnnealing {
    fn solve(&mut self, tsp: &TspLib) {
        let start_time = std::time::Instant::now();
        let mut rng = rand::thread_rng();
        let mut epoch = 0;

        let mut current_route = Route::new_random(&tsp.cities);
        let mut current_distance = current_route.distance;
        let mut best_distance = current_distance;
        self.best_route = current_route.clone();

        let moves_per_temp = tsp.dimension * 2;

        while self.temperature > self.min_temperature {
            if epoch % 1150 == 0 {
                println!(
                    "SA Epoch: {}, Temperature: {}, Best distance: {}",
                    epoch, self.temperature, best_distance
                );
            }

            for _ in 0..moves_per_temp {
                let new_route = current_route.random_move(&mut rng);
                let new_distance = new_route.distance;

                let delta = new_distance as f64 - current_distance as f64;
                let acceptance_probability = if delta < 0.0 {
                    1.0
                } else {
                    (-delta / self.temperature).exp()
                };

                if acceptance_probability > rng.gen::<f64>() {
                    current_route = new_route;
                    current_distance = new_distance;

                    if current_distance < best_distance {
                        best_distance = current_distance;
                        self.best_route = current_route.clone();
                    }
                }
            }

            self.history.push(self.best_route.clone());
            self.temperature *= 1.0 - self.cooling_rate;
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
