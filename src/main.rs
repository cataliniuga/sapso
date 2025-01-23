mod pso;
mod sa;
mod tsplib;

use anyhow::Result;

fn main() -> Result<()> {
    let arg = std::env::args()
        .nth(1)
        .expect("Usage: cargo run <tsp_instance_name>");
    let instance = format!("instances/{}.tsp", arg);

    let result = sa::solve_tsp(&instance)?;
    println!("Distance: {}", result.distance);

    Ok(())
}
