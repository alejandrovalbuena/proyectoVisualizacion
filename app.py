import requests
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html


app = Dash(__name__)


API_KEY = '51ZCQSNHYPT65QDB'
SYMBOL_AAPL = 'AAPL'
SYMBOL_AMZN = 'AMZN'

def fetch_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&datatype=json'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['Time Series (Daily)']).transpose()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'int'})
    return df


df_aapl = fetch_stock_data(SYMBOL_AAPL)


df_amzn = fetch_stock_data(SYMBOL_AMZN)


fig_aapl = go.Figure()
fig_aapl.add_trace(go.Scatter(x=df_aapl.index, y=df_aapl['Close'], mode='lines', name='AAPL Close Price'))


fig_amzn = go.Figure()
fig_amzn.add_trace(go.Scatter(x=df_amzn.index, y=df_amzn['Close'], mode='lines', name='AMZN Close Price'))


fig_comparison = go.Figure()
fig_comparison.add_trace(go.Scatter(x=df_aapl.index, y=df_aapl['Close'], mode='lines', name='AAPL Close Price'))
fig_comparison.add_trace(go.Scatter(x=df_amzn.index, y=df_amzn['Close'], mode='lines', name='AMZN Close Price'))


app.layout = html.Div(children=[
    html.H1(children='Stock Price Visualization'),
    dcc.Graph(id='apple-stock-price', figure=fig_aapl),
    dcc.Graph(id='amazon-stock-price', figure=fig_amzn),
    dcc.Graph(id='comparison-graph', figure=fig_comparison)
])

if __name__ == '__main__':
    app.run_server(debug=True)
