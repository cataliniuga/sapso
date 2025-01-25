use rand::{thread_rng, Rng};
use std::time::Instant;

use crate::tsplib::{City, HeuristicAlgorithm, Route, TspLib};

struct Particle {
    position: Vec<usize>,
    velocity: Vec<(usize, usize)>,
    best_position: Vec<usize>,
    best_fitness: u64,
}

impl Particle {
    fn new(num_cities: usize) -> Self {
        let position = (0..num_cities).collect::<Vec<usize>>();
        Particle {
            position: position.clone(),
            velocity: Vec::new(),
            best_position: position,
            best_fitness: u64::MAX,
        }
    }

    /// Initialize particle position using nearest neighbor heuristic
    fn initialize_nearest_neighbor(&mut self, distance_matrix: &[Vec<u64>]) {
        let mut rng = thread_rng();
        let mut current_city = rng.gen_range(0..self.position.len());
        let mut unvisited = (0..self.position.len())
            .filter(|&x| x != current_city)
            .collect::<Vec<usize>>();
        let mut route = vec![current_city];

        while !unvisited.is_empty() {
            let next_city = unvisited
                .iter()
                .min_by(|&&a, &&b| {
                    let dist_a = distance_matrix[current_city][a];
                    let dist_b = distance_matrix[current_city][b];
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .unwrap();
            let next_index = unvisited.iter().position(|&x| x == *next_city).unwrap();
            current_city = unvisited.remove(next_index);
            route.push(current_city);
        }

        self.position = route;
    }

    fn update_personal_best(&mut self, fitness: u64) {
        if fitness < self.best_fitness {
            self.best_fitness = fitness;
            self.best_position = self.position.clone();
        }
    }

    /// Order crossover (OX) operator
    fn crossover(&self, route1: &[usize], route2: &[usize]) -> Vec<usize> {
        let mut rng = thread_rng();
        let size = route1.len();
        let start = rng.gen_range(0..size);
        let end = rng.gen_range(start..size);

        // Get segment from first parent
        let segment = route1[start..=end].to_vec();

        // Create remaining elements in order of second parent
        let remaining = route2
            .iter()
            .filter(|x| !segment.contains(x))
            .copied()
            .collect::<Vec<usize>>();

        // Construct offspring
        let mut offspring = Vec::with_capacity(size);
        offspring.extend(&remaining[..start]);
        offspring.extend(&segment);
        offspring.extend(&remaining[start..]);

        offspring
    }

    /// Apply mutation (swap two random cities)
    fn mutate(&self, route: &mut [usize], mutation_rate: f64) {
        let mut rng = thread_rng();
        if rng.gen::<f64>() < mutation_rate {
            let i = rng.gen_range(0..route.len());
            let j = rng.gen_range(0..route.len());
            route.swap(i, j);
        }
    }

    /// Update particle's velocity using both PSO and genetic operators
    fn update_velocity(
        &mut self,
        cognitive_weight: f64,
        social_weight: f64,
        inertia_weight: f64,
        global_best_position: &[usize],
    ) {
        let mut rng = thread_rng();
        let mut new_route = self.position.clone();

        // Apply inertia: retain some previous swaps based on inertia weight
        let previous_swaps = self.velocity.clone();
        for swap in previous_swaps {
            if rng.gen::<f64>() < inertia_weight {
                let (i, j) = swap;
                new_route.swap(i, j);
            }
        }

        // PSO movement
        if rng.gen::<f64>() < cognitive_weight {
            new_route = self.crossover(&new_route, &self.best_position);
        }

        if rng.gen::<f64>() < social_weight {
            new_route = self.crossover(&new_route, global_best_position);
        }

        // Mutation
        self.mutate(&mut new_route, 0.1);

        // Convert differences into swap sequence
        self.velocity = self.get_swap_sequence(&new_route)
    }

    /// Generate sequence of swaps to transform from_route into to_route
    fn get_swap_sequence(&self, to_route: &[usize]) -> Vec<(usize, usize)> {
        let mut from_route = self.position.to_vec();
        let mut swaps = Vec::new();

        for i in 0..from_route.len() {
            if from_route[i] != to_route[i] {
                if let Some(j) = from_route.iter().position(|&x| x == to_route[i]) {
                    from_route.swap(i, j);
                    swaps.push((i, j));
                }
            }
        }

        swaps
    }

    /// Apply sequence of swaps to the route
    fn apply_velocity(&mut self) {
        for &(i, j) in self.velocity.iter() {
            self.position.swap(i, j);
        }
    }
}

/// Calculate total distance of the route
fn calculate_fitness(route: &[usize], distance_matrix: &[Vec<u64>]) -> u64 {
    let mut total_distance = 0;
    for i in 0..route.len() {
        let from_city = route[i];
        let to_city = route[(i + 1) % route.len()];
        total_distance += distance_matrix[from_city][to_city];
    }

    total_distance
}

pub struct ParticleSwarmOptimization {
    history: Vec<Route>,
    best_route: Route,
    run_time: u64,

    particles: Vec<Particle>,
    global_best_position: Vec<usize>,
    global_best_fitness: u64,
    max_iterations: usize,
    cognitive_weight: f64,
    social_weight: f64,
    inertia_weight: f64,
}

impl ParticleSwarmOptimization {
    pub fn new(
        tsp: &TspLib,
        num_particles: usize,
        max_iterations: usize,
        cognitive_weight: f64,
        social_weight: f64,
        inertia_weight: f64,
    ) -> Self {
        let mut particles = Vec::with_capacity(num_particles);
        let num_cities = tsp.dimension;
        let global_best_position = (0..num_cities).collect();

        // Initialize particles
        for _ in 0..num_particles {
            let mut particle = Particle::new(num_cities);
            particle.initialize_nearest_neighbor(&tsp.distance_matrix);
            particles.push(particle);
        }

        ParticleSwarmOptimization {
            history: Vec::new(),
            best_route: Route::new(&tsp.cities.clone()),
            run_time: 0,
            particles,
            global_best_position,
            global_best_fitness: u64::MAX,
            max_iterations,
            cognitive_weight,
            social_weight,
            inertia_weight,
        }
    }
}

impl HeuristicAlgorithm for ParticleSwarmOptimization {
    fn solve(&mut self, tsp: &TspLib) {
        let start_time = Instant::now();
        let mut current_best_fitness = self.global_best_fitness;

        // Initial evaluation
        for particle in &mut self.particles {
            let fitness = calculate_fitness(&particle.position, &tsp.distance_matrix);
            particle.update_personal_best(fitness);
            if fitness < self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best_position = particle.position.clone();
            }
        }

        for iteration in 0..self.max_iterations {
            for particle in &mut self.particles {
                particle.update_velocity(
                    self.cognitive_weight,
                    self.social_weight,
                    self.inertia_weight,
                    &self.global_best_position,
                );
                particle.apply_velocity();

                let fitness = calculate_fitness(&particle.position, &tsp.distance_matrix);

                particle.update_personal_best(fitness);

                if fitness < self.global_best_fitness {
                    self.global_best_fitness = fitness;
                    self.global_best_position = particle.position.clone();
                }
            }

            if self.global_best_fitness < current_best_fitness {
                current_best_fitness = self.global_best_fitness;
            }

            self.history.push(Route::new(
                &self
                    .global_best_position
                    .iter()
                    .map(|&city| tsp.cities[city])
                    .collect::<Vec<City>>(),
            ));

            if iteration % (self.max_iterations / 10) == 0 {
                println!(
                    "PSO Iteration {}/{}, Best distance: {}",
                    iteration, self.max_iterations, self.global_best_fitness
                );
            }
        }

        self.global_best_fitness =
            calculate_fitness(&self.global_best_position, &tsp.distance_matrix);

        self.best_route = Route::new(
            &self
                .global_best_position
                .iter()
                .map(|&city| tsp.cities[city])
                .collect::<Vec<City>>(),
        );
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
