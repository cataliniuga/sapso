use anyhow::Result;
use plotters::prelude::*;

use crate::sa::*;
use crate::tsplib::TspLib;

pub fn plot_sa(tsp: TspLib, sa: SimulatedAnnealing) -> Result<()> {
    let coord_range = tsp.node_coords.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |acc, &(x, y)| (acc.0.min(x), acc.1.max(x), acc.2.min(y), acc.3.max(y)),
    );

    // TSP PLOT
    let tsp_root = BitMapBackend::new("tsp.png", (2500, 1200)).into_drawing_area();
    tsp_root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&tsp_root)
        .caption("TSP Layout", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(
            coord_range.0 - 1.0..coord_range.1 + 1.0,
            coord_range.2 - 1.0..coord_range.3 + 1.0,
        )?;

    chart.configure_mesh().draw()?;

    let cities: Vec<(f64, f64)> = tsp.node_coords.iter().map(|&(x, y)| (x, y)).collect();
    chart.draw_series(PointSeries::of_element(
        cities.clone(),
        5,
        &BLACK,
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;
    tsp_root.present()?;

    // SIMMULATED ANNEALING PLOT
    let sa_root = BitMapBackend::new("sa.png", (2500, 1440)).into_drawing_area();
    sa_root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&sa_root)
        .caption("Simulated Annealing", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(
            coord_range.0 - 1.0..coord_range.1 + 1.0,
            coord_range.2 - 1.0..coord_range.3 + 1.0,
        )?;
    let best_route: Vec<(f64, f64)> = sa.best_route.cities.iter().map(|&(x, y)| (x, y)).collect();
    // char axes
    chart.configure_mesh().draw()?;
    // draw cities
    chart.draw_series(PointSeries::of_element(cities, 5, &BLACK, &|c, s, st| {
        EmptyElement::at(c) + Circle::new((0, 0), s, st.filled())
    }))?;
    chart.draw_series(LineSeries::new(best_route.clone(), &BLUE))?;
    // LAST -> FIRST
    chart.draw_series(LineSeries::new(
        vec![best_route[best_route.len() - 1], best_route[0]],
        &BLUE,
    ))?;
    sa_root.present()?;

    Ok(())
}
