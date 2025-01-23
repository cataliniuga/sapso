use crate::tsplib::TspLib;
use anyhow::Result;
use rand::{thread_rng, Rng};
use std::time::Instant;

#[derive(Clone)]
pub struct Particle {
    position: Vec<usize>,
    velocity: Vec<(usize, usize)>,
    best_position: Vec<usize>,
    best_fitness: f64,
}

impl Particle {
    fn new(num_cities: usize) -> Self {
        let position = (0..num_cities).collect::<Vec<usize>>();
        Particle {
            position: position.clone(),
            velocity: Vec::new(),
            best_position: position,
            best_fitness: f64::INFINITY,
        }
    }

    /// Initialize particle position using nearest neighbor heuristic
    fn initialize_nearest_neighbor(&mut self, distances: &[(f64, f64)]) {
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
                    let dist_a = euclidean_distance(&distances[current_city], &distances[a]);
                    let dist_b = euclidean_distance(&distances[current_city], &distances[b]);
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .unwrap();
            let next_index = unvisited.iter().position(|&x| x == *next_city).unwrap();
            current_city = unvisited.remove(next_index);
            route.push(current_city);
        }

        self.position = route;
    }

    fn update_personal_best(&mut self, fitness: f64) {
        if fitness < self.best_fitness {
            self.best_fitness = fitness;
            self.best_position = self.position.clone();
        }
    }
}

pub struct PSO {
    particles: Vec<Particle>,
    global_best_position: Vec<usize>,
    global_best_fitness: f64,
    num_particles: usize,
    max_iterations: usize,
    cognitive_weight: f64,
    social_weight: f64,
    inertia_weight: f64,
    local_search_freq: usize,
    node_coords: Vec<(f64, f64)>,
}

impl PSO {
    pub fn new(
        tsp: &TspLib,
        num_particles: usize,
        max_iterations: usize,
        cognitive_weight: f64,
        social_weight: f64,
        inertia_weight: f64,
        local_search_freq: usize,
    ) -> Self {
        let num_cities = tsp.dimension;
        let mut particles = Vec::with_capacity(num_particles);
        let mut global_best_fitness = f64::INFINITY;
        let global_best_position = (0..num_cities).collect();

        // Initialize particles
        for _ in 0..num_particles {
            let mut particle = Particle::new(num_cities);
            particle.initialize_nearest_neighbor(&tsp.node_coords);
            particles.push(particle);
        }

        PSO {
            particles,
            global_best_position,
            global_best_fitness,
            num_particles,
            max_iterations,
            cognitive_weight,
            social_weight,
            inertia_weight,
            local_search_freq,
            node_coords: tsp.node_coords.clone(),
        }
    }

    /// Calculate total distance of the route
    fn calculate_fitness(&self, route: &[usize]) -> f64 {
        let mut total_distance = 0.0;
        for i in 0..route.len() {
            let from_city = route[i];
            let to_city = route[(i + 1) % route.len()];
            total_distance +=
                euclidean_distance(&self.node_coords[from_city], &self.node_coords[to_city]);
        }

        total_distance
    }

    /// Perform 2-opt swap operation
    fn two_opt_swap(&self, route: &[usize], i: usize, j: usize) -> Vec<usize> {
        let mut new_route = route[..i].to_vec();
        new_route.extend(route[i..=j].iter().rev());
        new_route.extend(&route[j + 1..]);
        new_route
    }

    /// Apply randomized 2-opt local search to improve route
    fn local_search(&self, route: &[usize], max_iterations: usize) -> Vec<usize> {
        let mut rng = thread_rng();
        let mut best_route = route.to_vec();
        let mut best_distance = self.calculate_fitness(&best_route);
        let n = route.len();

        for _ in 0..max_iterations {
            // Randomly sample positions to swap
            let i = rng.gen_range(1..n - 2);
            let j = rng.gen_range(i + 1..n - 1);

            let new_route = self.two_opt_swap(&best_route, i, j);
            let new_distance = self.calculate_fitness(&new_route);

            if new_distance < best_distance {
                best_route = new_route;
                best_distance = new_distance;
            }
        }

        best_route
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
    fn mutate(&self, route: &mut Vec<usize>, mutation_rate: f64) {
        let mut rng = thread_rng();
        if rng.gen::<f64>() < mutation_rate {
            let i = rng.gen_range(0..route.len());
            let j = rng.gen_range(0..route.len());
            route.swap(i, j);
        }
    }

    /// Update particle's velocity using both PSO and genetic operators
    fn update_velocity(&self, particle: &Particle) -> Vec<(usize, usize)> {
        let mut rng = thread_rng();
        let mut new_route = particle.position.clone();

        // PSO movement
        if rng.gen::<f64>() < self.cognitive_weight {
            new_route = self.crossover(&new_route, &particle.best_position);
        }

        if rng.gen::<f64>() < self.social_weight {
            new_route = self.crossover(&new_route, &self.global_best_position);
        }

        // Mutation
        self.mutate(&mut new_route, 0.1);

        // Convert differences into swap sequence
        self.get_swap_sequence(&particle.position, &new_route)
    }

    /// Generate sequence of swaps to transform from_route into to_route
    fn get_swap_sequence(&self, from_route: &[usize], to_route: &[usize]) -> Vec<(usize, usize)> {
        let mut from_route = from_route.to_vec();
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
    fn apply_velocity(&self, route: &[usize], velocity: &[(usize, usize)]) -> Vec<usize> {
        let mut new_route = route.to_vec();
        for &(i, j) in velocity {
            new_route.swap(i, j);
        }
        new_route
    }

    /// Run the PSO algorithm
    pub fn optimize(&mut self) -> Result<(Vec<usize>, f64)> {
        let start_time = Instant::now();
        let mut iterations_without_improvement = 0;
        let mut current_best_fitness = self.global_best_fitness;

        // Initial evaluation
        for particle in &mut self.particles {
            let fitness = self.calculate_fitness(&particle.position);
            particle.update_personal_best(fitness);
            if fitness < self.global_best_fitness {
                self.global_best_fitness = fitness;
                self.global_best_position = particle.position.clone();
            }
        }

        for iteration in 0..self.max_iterations {
            let iteration_start = Instant::now();

            for particle in &mut self.particles {
                // Update velocity and position
                particle.velocity = self.update_velocity(particle);
                particle.position = self.apply_velocity(&particle.position, &particle.velocity);

                // Apply local search periodically
                if iteration % self.local_search_freq == 0 {
                    particle.position = self.local_search(&particle.position, 20);
                }

                // Evaluate new position
                let fitness = self.calculate_fitness(&particle.position);

                // Update personal best
                particle.update_personal_best(fitness);

                // Update global best
                if fitness < self.global_best_fitness {
                    self.global_best_fitness = fitness;
                    self.global_best_position = particle.position.clone();
                }
            }

            // Check improvement for this iteration
            if self.global_best_fitness < current_best_fitness {
                current_best_fitness = self.global_best_fitness;
                iterations_without_improvement = 0;
            } else {
                iterations_without_improvement += 1;
            }

            // Early stopping
            if iterations_without_improvement > 50 {
                println!("Early stopping at iteration {}", iteration);
                break;
            }

            println!(
                "Iteration {}/{}, Best distance: {:.2}, Time elapsed: {:.2}s",
                iteration + 1,
                self.max_iterations,
                self.global_best_fitness,
                iteration_start.elapsed().as_secs_f64()
            );
        }

        // Final local search on global best
        self.global_best_position = self.local_search(&self.global_best_position, 100);
        self.global_best_fitness = self.calculate_fitness(&self.global_best_position);

        println!(
            "Total time elapsed: {:.2}s",
            start_time.elapsed().as_secs_f64()
        );

        Ok((self.global_best_position.clone(), self.global_best_fitness))
    }
}

fn euclidean_distance(a: &(f64, f64), b: &(f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}
