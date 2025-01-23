use anyhow::Result;
use rand::{thread_rng, Rng};
use std::time::Instant;

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
        global_best_position: &[usize],
    ) {
        let mut rng = thread_rng();
        let mut new_route = self.position.clone();

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

/// Perform 2-opt swap operation
fn two_opt_swap(route: &[usize], i: usize, j: usize) -> Vec<usize> {
    let mut new_route = route[..i].to_vec();
    new_route.extend(route[i..=j].iter().rev());
    new_route.extend(&route[j + 1..]);
    new_route
}

/// Apply randomized 2-opt local search to improve route
fn local_search(
    route: &[usize],
    distance_matrix: &[Vec<u64>],
    max_iterations: usize,
) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut best_route = route.to_vec();
    let mut best_distance = calculate_fitness(&best_route, distance_matrix);
    let n = route.len();

    for _ in 0..max_iterations {
        // Randomly sample positions to swap
        let i = rng.gen_range(1..n - 2);
        let j = rng.gen_range(i + 1..n - 1);

        let new_route = two_opt_swap(&best_route, i, j);
        let new_distance = calculate_fitness(&new_route, distance_matrix);

        if new_distance < best_distance {
            best_route = new_route;
            best_distance = new_distance;
        }
    }

    best_route
}

struct PSO {
    particles: Vec<Particle>,
    global_best_position: Vec<usize>,
    global_best_fitness: u64,
    max_iterations: usize,
    cognitive_weight: f64,
    social_weight: f64,
    inertia_weight: f64,
    local_search_freq: usize,
}

impl PSO {
    fn new(
        distance_matrix: &[Vec<u64>],
        num_particles: usize,
        max_iterations: usize,
        cognitive_weight: f64,
        social_weight: f64,
        inertia_weight: f64,
        local_search_freq: usize,
    ) -> Self {
        let mut particles = Vec::with_capacity(num_particles);
        let num_cities = distance_matrix.len();
        let global_best_position = (0..num_cities).collect();

        // Initialize particles
        for _ in 0..num_particles {
            let mut particle = Particle::new(num_cities);
            particle.initialize_nearest_neighbor(distance_matrix);
            particles.push(particle);
        }

        PSO {
            particles,
            global_best_position,
            global_best_fitness: u64::MAX,
            max_iterations,
            cognitive_weight,
            social_weight,
            inertia_weight,
            local_search_freq,
        }
    }

    /// Run the PSO algorithm
    fn optimize(&mut self, distance_matrix: &[Vec<u64>]) -> Result<(Vec<usize>, u64, Vec<u64>)> {
        let start_time = Instant::now();
        let mut history = Vec::with_capacity(self.max_iterations);
        let mut iterations_without_improvement = 0;
        let mut current_best_fitness = self.global_best_fitness;

        // Initial evaluation
        for particle in &mut self.particles {
            let fitness = calculate_fitness(&particle.position, distance_matrix);
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
                particle.update_velocity(
                    self.cognitive_weight,
                    self.social_weight,
                    &self.global_best_position,
                );
                particle.apply_velocity();

                // Apply local search periodically
                if iteration % self.local_search_freq == 0 {
                    particle.position = local_search(&particle.position, distance_matrix, 20);
                }

                // Evaluate new position
                let fitness = calculate_fitness(&particle.position, distance_matrix);

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

            // Store history
            history.push(self.global_best_fitness);

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
        self.global_best_position = local_search(&self.global_best_position, distance_matrix, 100);
        self.global_best_fitness = calculate_fitness(&self.global_best_position, distance_matrix);

        println!(
            "Total time elapsed: {:.2}s",
            start_time.elapsed().as_secs_f64()
        );

        Ok((
            self.global_best_position.clone(),
            self.global_best_fitness,
            history,
        ))
    }
}

pub fn solve_tsp(distance_matrix: &[Vec<u64>]) -> Result<(Vec<usize>, u64, Vec<u64>)> {
    let mut pso = PSO::new(distance_matrix, 200, 1000, 1.5, 1.5, 0.8, 10);

    Ok(pso.optimize(distance_matrix).unwrap())
}
