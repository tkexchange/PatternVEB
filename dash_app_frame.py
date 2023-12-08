import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash import Dash
from dash.dependencies import Input, Output

import stock_pattern_analyzer as spa
from dash_app_functions import get_search_window_sizes, get_symbols, search_most_recent

app = Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}])
app.title = "Stock Patterns"
server = app.server

##### Settings container #####

symbol_dropdown_id = "id-symbol-dropdown"
available_symbols = get_symbols()
default_symbol = "AAPL" if "AAPL" in available_symbols else available_symbols[0]
symbol_dropdown = dcc.Dropdown(id=symbol_dropdown_id,
                               options=[{"label": x, "value": x} for x in available_symbols],
                               multi=False,
                               value=default_symbol,
                               className="dcc_control")

window_size_dropdown_id = "id-window-size-dropdown"
window_sizes = get_search_window_sizes()
window_size_dropdown = dcc.Dropdown(id=window_size_dropdown_id,
                                    options=[{"label": f"{x} days", "value": x} for x in window_sizes],
                                    multi=False,
                                    value=window_sizes[2],
                                    className="dcc_control")

future_size_input_id = "id-future-size-input"
MAX_FUTURE_WINDOW_SIZE = 10
future_size_input = dcc.Input(id=future_size_input_id, type="number", min=0, max=MAX_FUTURE_WINDOW_SIZE, value=5,
                              className="dcc_control")

top_k_input_id = "id-top-k-input"
MAX_TOP_K_VALUE = 10
top_k_input = dcc.Input(id=top_k_input_id, type="number", min=0, max=MAX_TOP_K_VALUE, value=5, className="dcc_control")

offset_checkbox_id = "id-offset-checkbox"
offset_checkbox = dcc.Checklist(id=offset_checkbox_id, options=[{"label": "Use Offset", "value": "offset"}],
                                value=["offset"], className="dcc_control")

settings_div = html.Div([html.P("Инструмент", className="control_label"),
                         symbol_dropdown,
                         html.P("Выбирите размер окна", className="control_label"),
                         window_size_dropdown,
                         html.P(f"Длина прогноза (дни мах. {MAX_FUTURE_WINDOW_SIZE})", className="control_label"),
                         future_size_input,
                         html.P(f"Количество прогнозов (max. {MAX_TOP_K_VALUE})", className="control_label"),
                         top_k_input,
                         offset_checkbox],
                        className="pretty_container three columns",
                        id="id-settings-div")

##### Stats & Graph #####

graph_id = "id-graph"
stats_and_graph_div = html.Div([html.Div(id="id-stats-container", className="row container-display"),
                                html.Div([dcc.Graph(id=graph_id)], id="id-graph-div", className="pretty_container")],
                               id="id-graph-container", className="nine columns")


##### Layout #####

app.layout = html.Div([
                       html.Div([settings_div,
                                 stats_and_graph_div],
                                className="row flex-display")],
                      id="mainContainer",
                      style={"display": "flex", "flex-direction": "column"})


##### Callbacks #####

@app.callback([Output(graph_id, "figure")],
              [Input(symbol_dropdown_id, "value"),
               Input(window_size_dropdown_id, "value"),
               Input(future_size_input_id, "value"),
               Input(top_k_input_id, "value"),
               Input(offset_checkbox_id, "value")])
def update_plot_and_table(symbol_value, window_size_value, future_size_value, top_k_value, checkbox_value):
    # RetAPI search
    ret = search_most_recent(symbol=symbol_value,
                             window_size=window_size_value,
                             top_k=top_k_value,
                             future_size=future_size_value)

    # Parse response and build the HTML table rows
    table_rows = []
    values = []
    symbols = []
    start_end_dates = []
    for i, match in enumerate(ret.matches):
        values.append(match.values)
        symbols.append(match.symbol)
        start_end_dates.append((match.start_date, match.end_date))
        row_values = [i + 1,
                      match.distance,
                      match.symbol,
                      match.end_date,
                      match.start_date,
                      match.values[-1],
                      match.values[window_size_value - 1],
                      match.values[0]]
        #row_dict = {c: v for c, v in zip(table_columns, row_values)}
        #table_rows.append(row_dict)

    offset_traces = False if len(checkbox_value) == 0 else True

    # Visualize the data on a graph
    fig = spa.visualize_graph(match_values_list=values,
                              match_symbols=symbols,
                              match_str_dates=start_end_dates,
                              window_size=window_size_value,
                              future_size=future_size_value,
                              anchor_symbol=ret.anchor_symbol,
                              anchor_values=ret.anchor_values,
                              show_legend=False,
                              offset_traces=offset_traces)

    return [fig]


if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0")
