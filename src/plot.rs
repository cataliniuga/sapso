use anyhow::Result;
use plotters::prelude::*;

use crate::tsplib::{HeuristicAlgorithm, Route, TspLib};

const FIG_SIZE: (u32, u32) = (2500, 1200);

pub fn plot_tsp_instance(tsp: TspLib) -> Result<()> {
    let coord_range = tsp.cities.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |acc, &(x, y)| (acc.0.min(x), acc.1.max(x), acc.2.min(y), acc.3.max(y)),
    );

    let tsp_root = BitMapBackend::new("./results/tsp.png", (2500, 1200)).into_drawing_area();
    tsp_root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&tsp_root)
        .caption("TSP Layout", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(75)
        .y_label_area_size(75)
        .build_cartesian_2d(
            coord_range.0 - 1.0..coord_range.1 + 1.0,
            coord_range.2 - 1.0..coord_range.3 + 1.0,
        )?;

    chart
        .configure_mesh()
        .x_desc("X")
        .y_desc("Y")
        .x_label_style(("sans-serif", 25).into_font())
        .y_label_style(("sans-serif", 25).into_font())
        .x_label_formatter(&|x| format!("{:.2}", x))
        .y_label_formatter(&|y| format!("{:.2}", y))
        .draw()?;

    chart.draw_series(PointSeries::of_element(
        tsp.cities.clone(),
        5,
        &BLACK,
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;
    if let Some(best_route) = tsp.optimal_tour {
        let best_route: Vec<(f64, f64)> = best_route.iter().map(|&i| tsp.cities[i]).collect();
        chart.draw_series(LineSeries::new(best_route.clone(), &RED))?;
        chart.draw_series(LineSeries::new(
            vec![best_route[best_route.len() - 1], best_route[0]],
            &RED,
        ))?;
    }
    tsp_root.present()?;

    Ok(())
}

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
    let coord_range = route.cities.iter().fold(
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ),
        |acc, &(x, y)| (acc.0.min(x), acc.1.max(x), acc.2.min(y), acc.3.max(y)),
    );

    let file_name = format!(
        "./results/{}_best_route.png",
        title.to_lowercase().replace(" ", "_")
    );
    let root = BitMapBackend::new(&file_name, FIG_SIZE).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(75)
        .y_label_area_size(75)
        .build_cartesian_2d(
            coord_range.0 - 1.0..coord_range.1 + 1.0,
            coord_range.2 - 1.0..coord_range.3 + 1.0,
        )?;

    chart
        .configure_mesh()
        .x_desc("X")
        .y_desc("Y")
        .x_label_style(("sans-serif", 25).into_font())
        .y_label_style(("sans-serif", 25).into_font())
        .x_label_formatter(&|x| format!("{:.2}", x))
        .y_label_formatter(&|y| format!("{:.2}", y))
        .draw()?;
    chart.draw_series(PointSeries::of_element(
        route.cities.clone(),
        5,
        &BLACK,
        &|c, s, st| EmptyElement::at(c) + Circle::new((0, 0), s, st.filled()),
    ))?;
    chart.draw_series(LineSeries::new(route.cities.clone(), color))?;
    chart.draw_series(LineSeries::new(
        vec![route.cities[route.cities.len() - 1], route.cities[0]],
        color,
    ))?;

    root.present()?;

    Ok(())
}

fn chart_history(history: Vec<Route>, title: &str) -> Result<()> {
    let file_name = format!(
        "./results/{}_history.png",
        title.to_lowercase().replace(" ", "_")
    );
    let root = BitMapBackend::new(&file_name, FIG_SIZE).into_drawing_area();
    root.fill(&WHITE)?;

    let min_distance = history.iter().map(|r| r.distance).min().unwrap_or(0);
    let max_distance = history.iter().map(|r| r.distance).max().unwrap_or(200);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(75)
        .y_label_area_size(75)
        .build_cartesian_2d(0..history.len() as u32, min_distance..max_distance)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Distance")
        .x_label_style(("sans-serif", 25).into_font())
        .y_label_style(("sans-serif", 25).into_font())
        .draw()?;

    chart.draw_series(LineSeries::new(
        history
            .iter()
            .enumerate()
            .map(|(i, r)| (i as u32, r.distance)),
        &RED,
    ))?;

    Ok(())
}
