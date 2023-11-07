import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, callback, State, Input, Output
from dash.exceptions import PreventUpdate
import copy
from dash.dependencies import Input, Output


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plots.plots import (
    plot_attention_from_checkpoint,
    plot_embeddings_from_checkpoint,
    plot_vectors,
    plot_sat_curves_from_parquet,
    plot_vectors_multiple_epochs,
)

from utils.dash_paths import (
    get_checkpoint_path,
    checkpoint_output_results_path,
    batch_parquet_file_path,
)


dash.register_page(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Control panel layout
control_panel_layout = dbc.Card(
    [
        dbc.CardHeader(html.H4("Control Panel", className="card-title")),
        dbc.CardBody(
            [
                html.P("Select Model", className="card-text fw-bold mb-2"),
                dcc.Dropdown(
                    id="model-dropdown",
                    options=[
                        {"label": model, "value": model}
                        for model in [
                            "ckpt",
                            "flesch_kincaid_asc",
                            "flesch_kincaid_desc",
                        ]
                    ],
                    value="flesch_kincaid_asc",
                    clearable=False,
                    style={"margin-bottom": "10px"},
                ),
                html.P("Select Plot Type", className="card-text fw-bold mb-2"),
                dcc.Dropdown(
                    id="plot-type-dropdown",
                    options=[
                        {"label": "Layer Similarity", "value": "vector-layer"},
                        {"label": "Attention Weights", "value": "attention"},
                        {"label": "Embedding Visualization", "value": "embedding"},
                        # Add other plot types here
                    ],
                    value="vector-layer",
                    clearable=False,
                    style={"margin-bottom": "10px"},
                ),
                html.P("Select Epoch(s)", className="card-text fw-bold mb-2"),
                dcc.Dropdown(
                    id="epoch-dropdown",
                    options=[
                        {"label": f"Epoch {i}", "value": i}
                        for i in range(100, 901, 100)
                    ],
                    value=[
                        100
                    ],  # Default to the first epoch, but allow multiple selections
                    multi=True,  # Allow multiple selections
                    clearable=False,
                    style={"margin-bottom": "10px"},
                ),
            ],
            className="mb-3",
            style={"maxWidth": "24rem"},
        ),
    ],
    className="mb-3",
)

# Main layout of the page
layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1("Model Analysis Dashboard"),
                    width=12,
                    className="text-center bg-primary text-white p-4 mb-4",
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(control_panel_layout, md=3),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Loading(
                                        children=html.Div(id="sat-curves-plot")
                                    ),
                                    md=12,
                                ),
                            ],
                            className="mb-4",  # Add margin below the SAT curve plots
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Loading(children=html.Div(id="selected-plot")),
                                    md=12,  # This should span the full width of the right column
                                ),
                            ],
                        ),
                    ],
                    md=9,
                ),
            ]
        ),
    ],
    fluid=True,
)


# Callbacks for plotting SAT curves output and batch results
@callback(Output("sat-curves-plot", "children"), [Input("model-dropdown", "value")])
def update_sat_curves_plot(model_name):
    output_path = checkpoint_output_results_path(model_name)
    batch_path = batch_parquet_file_path(model_name)
    fig = plot_sat_curves_from_parquet(output_path, batch_path, "SAT Curves")
    return dcc.Graph(figure=fig)


@callback(
    Output("selected-plot", "children"),
    [
        Input("model-dropdown", "value"),
        Input("plot-type-dropdown", "value"),
        Input("epoch-dropdown", "value"),
    ],
)
def update_selected_plot(model_name, plot_type, epochs):
    # Build a list of checkpoint paths based on selected epochs
    checkpoint_paths = [get_checkpoint_path(model_name, epoch) for epoch in epochs]

    # Handle the case where multiple epochs are selected
    if len(epochs) > 1:
        if plot_type == "vector-layer":
            # Use the function for plotting across multiple epochs
            fig = plot_vectors_multiple_epochs(checkpoint_paths)
            return dcc.Graph(figure=fig)
        else:
            # Placeholder for other plot types, if needed
            return html.Div(
                "Multiple epoch comparison plot for the selected plot type to be implemented."
            )
    else:
        # Handle the case for a single epoch
        checkpoint_path = checkpoint_paths[
            0
        ]  # There will be only one path in this case

        if plot_type == "attention":
            fig = plot_attention_from_checkpoint(checkpoint_path)
        elif plot_type == "embedding":
            fig = plot_embeddings_from_checkpoint(checkpoint_path)
        elif plot_type == "vector-layer":
            fig = plot_vectors(checkpoint_path)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        return (
            dcc.Graph(figure=fig)
            if fig is not None
            else html.Div("No data available for the selected model and plot type.")
        )
