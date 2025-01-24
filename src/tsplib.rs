use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufRead, BufReader},
    vec,
};

use anyhow::Result;
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};

static OPTIMALS_PATH: &str = "instances/optimal_tour_lengths.txt";

fn euclidean_distance(a: &City, b: &City) -> u64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    let distance = (dx * dx + dy * dy).sqrt();

    distance.round() as u64
}

pub type City = (f64, f64);

#[derive(Clone)]
pub struct Route {
    pub cities: Vec<City>,
    pub distance: u64,
}

impl Route {
    pub fn new(coords: &[City]) -> Self {
        let cities: Vec<City> = coords.iter().map(|&(x, y)| (x, y)).collect();
        let distance = Self::calculate_distance(&cities);
        Route { cities, distance }
    }

    pub fn new_random(coords: &[City]) -> Self {
        let mut cities: Vec<City> = coords.iter().map(|&(x, y)| (x, y)).collect();
        let mut rng = rand::thread_rng();
        cities.shuffle(&mut rng);
        let distance = Self::calculate_distance(&cities);
        Route { cities, distance }
    }

    pub fn calculate_distance(cities: &[City]) -> u64 {
        let mut distance = euclidean_distance(&cities[cities.len() - 1], &cities[0]);
        for i in 1..cities.len() {
            distance += euclidean_distance(&cities[i - 1], &cities[i]);
        }
        distance
    }

    fn swap_random_cities(&self, rng: &mut rand::prelude::ThreadRng) -> Self {
        let mut new_cities = self.cities.clone();
        let i = rng.gen_range(0..new_cities.len());
        let j = rng.gen_range(0..new_cities.len());
        new_cities.swap(i, j);
        let distance = Self::calculate_distance(&new_cities);
        Route {
            cities: new_cities,
            distance,
        }
    }

    pub fn two_opt_move(&self, i: usize, j: usize) -> Self {
        let mut new_cities = self.cities.clone();

        let (left, right) = (i.min(j), i.max(j));
        new_cities[left..=right].reverse();

        let distance = Self::calculate_distance(&new_cities);
        Route {
            cities: new_cities,
            distance,
        }
    }

    pub fn random_move(&self, rng: &mut ThreadRng) -> Self {
        if rng.gen::<f64>() < 0.8 {
            self.swap_random_cities(rng)
        } else {
            let i = rng.gen_range(0..self.cities.len());
            let j = rng.gen_range(0..self.cities.len());
            self.two_opt_move(i, j)
        }
    }
}

pub trait HeuristicAlgorithm {
    fn solve(&mut self, tsp: &TspLib);
    fn get_history(&self) -> Vec<Route>;
    fn get_best_route(&self) -> Route;
    fn get_run_time(&self) -> u64;
}

#[derive(Clone)]
pub struct TspLib {
    pub name: String,
    pub comment: String,
    pub dimension: usize,
    pub cities: Vec<City>,
    pub distance_matrix: Vec<Vec<u64>>,
    pub optimal_tour: Option<Vec<usize>>,
    pub optimal_tour_length: Option<u64>,
}

impl TspLib {
    pub fn new() -> TspLib {
        TspLib {
            name: String::new(),
            comment: String::new(),
            dimension: 0,
            cities: Vec::new(),
            distance_matrix: Vec::new(),
            optimal_tour: None,
            optimal_tour_length: None,
        }
    }
}

impl std::fmt::Debug for TspLib {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let optimal_tour_length = match self.optimal_tour_length {
            Some(length) => length.to_string(),
            None => "None".to_string(),
        };
        write!(
            f,
            "TspLib {{ name: {}, comment: {}, dimension: {}, optimal_tour_length: {:?} }}",
            self.name, self.comment, self.dimension, optimal_tour_length
        )
    }
}

pub fn get_optimal_tour_length() -> Result<HashMap<String, u64>> {
    let file = File::open(OPTIMALS_PATH)?;
    let reader = BufReader::new(file);

    let mut optimal_tour_lengths = HashMap::new();
    for line in reader.lines() {
        let line = line?;
        let parts = line.split_whitespace().collect::<Vec<&str>>();
        let name = parts[0].to_string();
        let length = parts[1].parse()?;
        optimal_tour_lengths.insert(name, length);
    }

    Ok(optimal_tour_lengths)
}

pub fn read_tsp_file(filename: &str) -> Result<TspLib> {
    let mut tsp = TspLib::new();
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let mut line = lines.next().unwrap()?;

    assert!(line.contains("NAME"));
    tsp.name = line.split(":").collect::<Vec<&str>>()[1].trim().to_string();
    line = lines.next().unwrap()?;

    while !line.contains("NODE_COORD_SECTION") {
        if line.contains("NAME") {
            tsp.name = line.split(":").collect::<Vec<&str>>()[1].trim().to_string();
        } else if line.contains("COMMENT") {
            tsp.comment = line.split(":").collect::<Vec<&str>>()[1].trim().to_string();
        } else if line.contains("DIMENSION") {
            tsp.dimension = line.split(":").collect::<Vec<&str>>()[1].trim().parse()?;
        } else if line.contains("EDGE_WEIGHT_TYPE") {
            let edge_weight_type = line.split(":").collect::<Vec<&str>>()[1].trim();
            assert_eq!(edge_weight_type, "EUC_2D");
        }
        line = lines.next().unwrap()?;
    }

    for _ in 0..tsp.dimension {
        line = lines.next().unwrap()?;
        let coords = line.split_whitespace().collect::<Vec<&str>>();
        let x = coords[1].parse()?;
        let y = coords[2].parse()?;
        tsp.cities.push((x, y));
    }

    tsp.distance_matrix = vec![vec![0; tsp.dimension]; tsp.dimension];
    for i in 0..tsp.dimension - 1 {
        for j in i + 1..tsp.dimension {
            let dist = euclidean_distance(&tsp.cities[i], &tsp.cities[j]);
            tsp.distance_matrix[i][j] = dist;
            tsp.distance_matrix[j][i] = dist;
        }
    }

    if fs::exists(format!("instances/{}.opt.tour", tsp.name))? {
        let file = File::open(format!("instances/{}.opt.tour", tsp.name))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        while !line.contains("TOUR_SECTION") {
            line = lines.next().unwrap()?;
        }
        let mut optimal_tour = Vec::new();
        for _ in 0..tsp.dimension {
            line = lines.next().unwrap()?;
            if line.contains("-1") {
                break;
            }
            let node = line.trim().parse::<usize>()?;
            optimal_tour.push(node - 1);
        }
        tsp.optimal_tour = Some(optimal_tour);
    }

    let optimal_tour_lengths = get_optimal_tour_length()?;
    if let Some(&length) = optimal_tour_lengths.get(&tsp.name) {
        tsp.optimal_tour_length = Some(length);
    }

    Ok(tsp)
}
