use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use anyhow::Result;

pub struct TspLib {
    pub name: String,
    pub comment: String,
    pub dimension: usize,
    pub node_coords: Vec<(f64, f64)>,
}

impl TspLib {
    pub fn new() -> TspLib {
        TspLib {
            name: String::new(),
            comment: String::new(),
            dimension: 0,
            node_coords: Vec::new(),
        }
    }
}

pub fn read_tsp_file(filename: &str) -> Result<TspLib> {
    let mut tsp = TspLib::new();
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let mut line = lines.next().unwrap()?;
    while !line.contains("NODE_COORD_SECTION") {
        if line.contains("NAME") {
            tsp.name = line.split(":").collect::<Vec<&str>>()[1].trim().to_string();
        } else if line.contains("COMMENT") {
            tsp.comment = line.split(":").collect::<Vec<&str>>()[1].trim().to_string();
        } else if line.contains("DIMENSION") {
            tsp.dimension = line.split(":").collect::<Vec<&str>>()[1].trim().parse()?;
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

    Ok(tsp)
}
