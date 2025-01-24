use std::{
    collections::HashMap,
    fmt::format,
    fs::{self, File},
    io::{BufRead, BufReader},
    vec,
};

use anyhow::Result;

static OPTIMALS_PATH: &str = "instances/optimal_tour_lengths.txt";

fn euclidean_distance(a: &(f64, f64), b: &(f64, f64)) -> u64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt() as u64
}

#[derive(Clone)]
pub struct TspLib {
    pub name: String,
    pub comment: String,
    pub dimension: usize,
    pub node_coords: Vec<(f64, f64)>,
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
            node_coords: Vec::new(),
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
        tsp.node_coords.push((x, y));
    }

    tsp.distance_matrix = vec![vec![0; tsp.dimension]; tsp.dimension];
    for i in 0..tsp.dimension - 1 {
        for j in i + 1..tsp.dimension {
            let dist = euclidean_distance(&tsp.node_coords[i], &tsp.node_coords[j]);
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
