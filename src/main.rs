mod plot;
mod pso;
mod sa;
mod tsplib;

use anyhow::Result;
use tsplib::read_tsp_file;

fn main() -> Result<()> {
    let arg = std::env::args().nth(1).unwrap_or("a280".to_string());
    let instance = format!("instances/{}.tsp", arg);
    let tsp = read_tsp_file(&instance)?;

    println!("{:?}", tsp);

    let sa_result = sa::solve_tsp(tsp.clone())?;
    println!("SA best distance: {}", sa_result.best_route.distance);

    let pso_result = pso::solve_tsp(&tsp.distance_matrix)?;
    println!("PSO best distance: {}", pso_result.1);

    plot::plot_sa(tsp, sa_result)?;

    Ok(())
}
