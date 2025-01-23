use std::{
    fs::File,
    io::{BufRead, BufReader}, vec,
};

use anyhow::Result;

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
}

impl TspLib {
    pub fn new() -> TspLib {
        TspLib {
            name: String::new(),
            comment: String::new(),
            dimension: 0,
            node_coords: Vec::new(),
            distance_matrix: Vec::new(),
        }
    }
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

    assert!(line.contains("COMMENT"));
    tsp.comment = line.split(":").collect::<Vec<&str>>()[1].trim().to_string();
    line = lines.next().unwrap()?;

    assert!(line.contains("TYPE"));
    line = lines.next().unwrap()?;

    assert!(line.contains("DIMENSION"));
    tsp.dimension = line.split(":").collect::<Vec<&str>>()[1].trim().parse()?;
    line = lines.next().unwrap()?;

    assert!(line.contains("EDGE_WEIGHT_TYPE"));
    line = lines.next().unwrap()?;
    
    assert!(line.contains("NODE_COORD_SECTION"));

    for _ in 0..tsp.dimension {
        line = lines.next().unwrap()?;
        let coords = line.split_whitespace().collect::<Vec<&str>>();
        let x = coords[1].parse()?;
        let y = coords[2].parse()?;
        tsp.node_coords.push((x, y));
    }

    tsp.distance_matrix = vec![vec![0; tsp.dimension]; tsp.dimension];
    for i in 0..tsp.dimension-1 {
        for j in i+1..tsp.dimension {
            let dist = euclidean_distance(&tsp.node_coords[i], &tsp.node_coords[j]);
            tsp.distance_matrix[i][j] = dist;
            tsp.distance_matrix[j][i] = dist;
        }
    }

    Ok(tsp)
}
