use crate::tsplib::*;
use anyhow::Result;
use rand::prelude::*;
use std::f64::consts::E;

#[derive(Clone)]
pub struct City {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone)]
pub struct Route {
    pub cities: Vec<City>,
    pub distance: f64,
}

impl Route {
    pub fn new(coords: &[(f64, f64)]) -> Self {
        let cities: Vec<City> = coords.iter().map(|&(x, y)| City { x, y }).collect();
        let distance = Self::calculate_distance(&cities);
        Route { cities, distance }
    }

    fn calculate_distance(cities: &[City]) -> f64 {
        let mut distance = euclidian_distance(&cities[cities.len() - 1], &cities[0]);
        for i in 1..cities.len() {
            distance += euclidian_distance(&cities[i - 1], &cities[i]);
        }
        distance
    }

    fn generate_neighbor(&self) -> Self {
        let mut rng = thread_rng();
        let len = self.cities.len();
        let mut new_cities = self.cities.clone();

        let i = rng.gen_range(0..len);
        let mut j = rng.gen_range(0..len);
        while i == j {
            j = rng.gen_range(0..len);
        }
        new_cities.swap(i, j);

        let distance = Self::calculate_distance(&new_cities);
        Route {
            cities: new_cities,
            distance,
        }
    }
}

fn euclidian_distance(a: &City, b: &City) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

pub struct SimulatedAnnealing {
    pub temperature: f64,
    pub cooling_rate: f64,
    pub min_temperature: f64,
}

impl SimulatedAnnealing {
    pub fn new(temperature: f64, cooling_rate: f64, min_temperature: f64) -> Self {
        SimulatedAnnealing {
            temperature,
            cooling_rate,
            min_temperature,
        }
    }

    pub fn solve(&mut self, tsp: &TspLib) -> Route {
        let mut rng = thread_rng();
        let mut current_route = Route::new(&tsp.node_coords);
        let mut best_route = current_route.clone();

        while self.temperature > self.min_temperature {
            let neighbor = current_route.generate_neighbor();
            let delta = neighbor.distance - current_route.distance;

            if delta < 0.0 || rng.gen::<f64>() < E.powf(-delta / self.temperature) {
                current_route = neighbor;
                if current_route.distance < best_route.distance {
                    best_route = current_route.clone();
                }
            }

            self.temperature *= 1.0 - self.cooling_rate;
        }

        best_route
    }
}

pub fn solve_tsp(filename: &str) -> Result<Route> {
    let tsp = read_tsp_file(filename)?;

    let mut sa = SimulatedAnnealing::new(100.0, 0.001, 0.01);

    Ok(sa.solve(&tsp))
}
