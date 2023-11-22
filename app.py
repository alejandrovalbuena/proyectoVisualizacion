import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import requests


app = dash.Dash(__name__)

API_KEY = '51ZCQSNHYPT65QDB'

top_50_sp500 = [
    {'label': 'Apple Inc', 'value': 'AAPL'},
    {'label': 'Microsoft Corp', 'value': 'MSFT'},
    {'label': 'Amazon.com Inc', 'value': 'AMZN'},
    {'label': 'Facebook Inc', 'value': 'FB'},
    {'label': 'Alphabet Inc (Google)', 'value': 'GOOGL'}
]



def fetch_stock_data(symbol):
    API_KEY = 'YourAlphaVantageAPIKey'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&datatype=json'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['Time Series (Daily)']).transpose()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'int'})
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
