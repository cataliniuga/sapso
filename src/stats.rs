use crate::sa;
use crate::tsplib::TspLib;

pub fn get_history_distances(sa: sa::SimulatedAnnealing) -> Vec<f64> {
    sa.history.iter().map(|r| r.distance).collect()
}

