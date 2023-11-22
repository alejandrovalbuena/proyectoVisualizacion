import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import yfinance as yf

app = dash.Dash(__name__)

top_50_sp500 = [
    {'label': 'Apple Inc', 'value': 'AAPL'},
    {'label': 'Microsoft Corp', 'value': 'MSFT'},
    {'label': 'Amazon.com Inc', 'value': 'AMZN'},
    {'label': 'Facebook Inc', 'value': 'FB'},
    {'label': 'Alphabet Inc (Google)', 'value': 'GOOGL'}
]

def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    df = stock.history(period="1y")  # You can change the period as needed
    return df

app.layout = html.Div(children=[
    html.H1(children='S&P 500 Stock App'),

    dcc.Dropdown(
        id='stock-selector',
        options=top_50_sp500,
        value='AAPL'  
    ),

    dcc.Graph(id='stock-price-graph')
])

@app.callback(
    Output('stock-price-graph', 'figure'),
    [Input('stock-selector', 'value')]
)
def update_graph(selected_stock):
    df_stock = fetch_stock_data(selected_stock)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], mode='lines', name=f'{selected_stock} Close Price'))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
