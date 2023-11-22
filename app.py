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
    {'label': 'Meta Platforms', 'value': 'META'},
    {'label': 'Alphabet Inc (Google)', 'value': 'GOOGL'}
]

def fetch_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

app.layout = html.Div(children=[
    html.H1(children='S&P 500 Stock App'),

    dcc.Dropdown(
        id='stock-selector',
        options=top_50_sp500,
        value='AAPL'
    ),

    dcc.Dropdown(
        id='time-range-selector',
        options=[
            {'label': '1 Month', 'value': '1mo'},
            {'label': '3 Months', 'value': '3mo'},
            {'label': '6 Months', 'value': '6mo'},
            {'label': '1 Year', 'value': '1y'},
            {'label': '5 Years', 'value': '5y'}
        ],
        value='1y'
    ),

    dcc.Checklist(
        id='ma-selector',
        options=[{'label': 'Show 200-session Moving Average (Only in 1 and 5 year time frames)', 'value': 'MA200'}],
        value=[]
    ),

    dcc.Graph(id='stock-price-graph')
])

@app.callback(
    Output('stock-price-graph', 'figure'),
    [Input('stock-selector', 'value'), Input('ma-selector', 'value'), Input('time-range-selector', 'value')]
)
def update_graph(selected_stock, ma_options, selected_time_range):
    df_stock = fetch_stock_data(selected_stock, selected_time_range)
    df_stock.sort_index(inplace=True)

    fig = go.Figure()

    # Plotting the close price
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], mode='lines', name=f'{selected_stock} Close Price'))

    # Check if the 200-session moving average is selected and the time range is either 1 year or 5 years
    if 'MA200' in ma_options and selected_time_range in ['1y', '5y']:
        # Calculate 200-session moving average
        # Using min_periods=1 will start the MA line from the beginning of the dataset
        df_stock['200_MA'] = df_stock['Close'].rolling(window=200, min_periods=1).mean()
        
        # Plotting the 200-session moving average
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['200_MA'], mode='lines', name='200 Session MA', line=dict(color='red')))

    return fig



if __name__ == '__main__':
    app.run_server(debug=True)
