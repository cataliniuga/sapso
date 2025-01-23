mod pso;
mod sa;
mod tsplib;

use anyhow::Result;
use tsplib::read_tsp_file;

fn main() -> Result<()> {
    let arg = std::env::args()
        .nth(1)
        .expect("Usage: cargo run <tsp_instance_name>");
    let instance = format!("instances/{}.tsp", arg);
    let tsp = read_tsp_file(&instance)?;

    let sa_result = sa::solve_tsp(tsp.clone())?;
    println!("Distance: {}", sa_result.distance);

    let pso_result = pso::solve_tsp(&tsp.distance_matrix)?;
    println!("Total distance: {}", pso_result.1);

    Ok(())
}
