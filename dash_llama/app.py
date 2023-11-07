import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

dropdown_menu = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem(f"{page['name']}", href=page["relative_path"])
        for page in dash.page_registry.values()
    ],
    nav=True,
    in_navbar=True,
    label="Pages",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Prebiasing LLMs", href="/"),
            dbc.Nav(
                dropdown_menu,
                className="ms-auto",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
)

app.layout = html.Div(
    [
        navbar,
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
