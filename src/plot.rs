use anyhow::Result;
use plotters::prelude::*;

use crate::tsplib::{HeuristicAlgorithm, Route};

pub fn plot_algo_result(
    ha: &dyn HeuristicAlgorithm,
    title: &str,
    color: &plotters::style::RGBColor,
) -> Result<()> {
    plot_alg_best_route(ha.get_best_route(), title, color)?;
    chart_history(ha.get_history(), title)?;

    Ok(())
}

fn plot_alg_best_route(route: Route, title: &str, color: &plotters::style::RGBColor) -> Result<()> {
    let file_name = format!("results/{}.png", title.to_lowercase().replace(" ", "_"));
    let root = BitMapBackend::new(&file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..route.cities.len() as u32, 0..route.distance)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        route.cities.iter().map(|c| (c.0 as u32, c.1 as u64)),
        color,
    ))?;

    Ok(())
}

fn chart_history(history: Vec<Route>, title: &str) -> Result<()> {
    let file_name = format!("results/{}.png", title);
    let root = BitMapBackend::new(&file_name, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..history.len() as u32, 0..history[0].distance)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        history
            .iter()
            .enumerate()
            .map(|(i, r)| (i as u32, r.distance)),
        &RED,
    ))?;

    Ok(())
}
