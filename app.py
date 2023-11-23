import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import linregress
from plotly.subplots import make_subplots
from datetime import datetime

app = dash.Dash(__name__)

top_50_sp500 = [
    {'label': 'Apple Inc', 'value': 'AAPL'},
    {'label': 'Microsoft Corp', 'value': 'MSFT'},
    {'label': 'Amazon.com Inc', 'value': 'AMZN'},
    {'label': 'Meta Platforms', 'value': 'META'},
    {'label': 'Alphabet Inc (Google)', 'value': 'GOOGL'},
    {'label': 'Berkshire Hathaway', 'value': 'BRK.B'},
    {'label': 'Johnson & Johnson', 'value': 'JNJ'},
    {'label': 'JPMorgan Chase & Co.', 'value': 'JPM'},
    {'label': 'Visa Inc', 'value': 'V'},
    {'label': 'Procter & Gamble Co', 'value': 'PG'},
    {'label': 'UnitedHealth Group', 'value': 'UNH'},
    {'label': 'NVIDIA Corporation', 'value': 'NVDA'},
    {'label': 'Home Depot', 'value': 'HD'},
    {'label': 'Tesla Inc', 'value': 'TSLA'},
    {'label': 'Mastercard Inc', 'value': 'MA'},
    {'label': 'Walt Disney Co', 'value': 'DIS'},
    {'label': 'PayPal Holdings', 'value': 'PYPL'},
    {'label': 'Comcast Corp', 'value': 'CMCSA'},
    {'label': 'Adobe Inc', 'value': 'ADBE'},
    {'label': 'Netflix Inc', 'value': 'NFLX'},
    {'label': 'Intel Corp', 'value': 'INTC'},
    {'label': 'Verizon Communications', 'value': 'VZ'},
    {'label': 'Coca-Cola Co', 'value': 'KO'},
    {'label': 'AT&T Inc', 'value': 'T'},
    {'label': 'Pfizer Inc', 'value': 'PFE'},
    {'label': 'Cisco Systems', 'value': 'CSCO'},
    {'label': 'PepsiCo Inc', 'value': 'PEP'},
    {'label': 'Merck & Co Inc', 'value': 'MRK'},
    {'label': 'Walmart Inc', 'value': 'WMT'},
    {'label': 'Broadcom Inc', 'value': 'AVGO'}
]


def fetch_stock_data(symbol, period):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

def calculate_rsi(dataframe, window=14):
    delta = dataframe['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window).mean()
    ma_down = down.rolling(window).mean()
    
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    rsi = rsi.rename("RSI")
    return rsi

def calculate_bollinger_bands(dataframe, window=20, num_of_std=2):
    rolling_mean = dataframe['Close'].rolling(window=window).mean()
    rolling_std = dataframe['Close'].rolling(window=window).std()
    
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    
    return rolling_mean, upper_band, lower_band

app.layout = html.Div([
    html.H1('S&P 500 Stock App'),

    dcc.Dropdown(
        id='stock-selector',
        options=top_50_sp500,
        value='AAPL'  # Default value to Apple Inc
    ),

    html.Button('Select All Indicators', id='select-all-button'),

    dcc.Dropdown(
        id='time-range-selector',
        options=[
            {'label': '1 Month', 'value': '1mo'},
            {'label': '3 Months', 'value': '3mo'},
            {'label': '6 Months', 'value': '6mo'},
            {'label': '1 Year', 'value': '1y'},
            {'label': '5 Years', 'value': '5y'}
        ],
        value='1y'  # Default value to 1 Year
    ),

    html.Div([
        dcc.Checklist(
            id='ma-options-selector',
            options=[
                {'label': 'Show 200-session Moving Average', 'value': 'MA200'},
                {'label': 'Show 100-session Moving Average', 'value': 'MA100'},
                {'label': 'Show 50-session Moving Average', 'value': 'MA50'}
            ],
            value=['MA200']  # Default value with 200-session MA selected
        ),
        dcc.Graph(id='stock-price-graph')
    ]),

    html.Div([
        dcc.Checklist(
            id='indicator-selector',
            options=[
                {'label': 'Show RSI', 'value': 'RSI'},
                {'label': 'Show Bollinger Bands', 'value': 'BOLL'}
            ],
            value=[]  # No default value, user must select
        ),
        dcc.Graph(id='indicator-graph')
    ])
])

@app.callback(
    Output('ma-options-selector', 'value'),
    Output('indicator-selector', 'value'),
    Input('select-all-button', 'n_clicks'),
    State('ma-options-selector', 'options'),
    State('indicator-selector', 'options'),
    prevent_initial_call=True
)
def select_all(n_clicks, ma_options, indicator_options):
    if n_clicks is None:
        raise PreventUpdate

    # Extract the values from options
    ma_values = [option['value'] for option in ma_options]
    indicator_values = [option['value'] for option in indicator_options]

    return ma_values, indicator_values

@app.callback(
    Output('stock-price-graph', 'figure'),
    [Input('stock-selector', 'value'), Input('ma-options-selector', 'value'), Input('time-range-selector', 'value')]
)
def update_graph(selected_stock, ma_options, selected_time_range):
    df_stock = fetch_stock_data(selected_stock, selected_time_range)
    fig = go.Figure()

    # Plotting the close price
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], mode='lines', name=f'{selected_stock} Close Price'))

    # Calculate and plot 200-session moving average if selected
    if 'MA200' in ma_options:
        df_stock['200_MA'] = df_stock['Close'].rolling(window=200, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['200_MA'], mode='lines', name='200 Session MA', line=dict(color='red')))

    # Calculate and plot 100-session moving average if selected
    if 'MA100' in ma_options:
        df_stock['100_MA'] = df_stock['Close'].rolling(window=100, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['100_MA'], mode='lines', name='100 Session MA', line=dict(color='green')))

    # Calculate and plot 50-session moving average if selected
    if 'MA50' in ma_options:
        df_stock['50_MA'] = df_stock['Close'].rolling(window=50, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['50_MA'], mode='lines', name='50 Session MA', line=dict(color='orange')))

    return fig


@app.callback(
    Output('indicator-graph', 'figure'),
    [Input('stock-selector', 'value'),
     Input('indicator-selector', 'value'),
     Input('time-range-selector', 'value')]
)
def update_indicator_graph(selected_stock, selected_indicators, selected_time_range):
    df_stock = fetch_stock_data(selected_stock, selected_time_range)
    
    if df_stock.empty:
        return go.Figure()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if 'RSI' in selected_indicators:
        rsi = calculate_rsi(df_stock)
        fig.add_trace(
            go.Scatter(x=df_stock.index, y=rsi, mode='lines', name='RSI'),
            secondary_y=False,
        )

    if 'BOLL' in selected_indicators:
        rolling_mean, upper_band, lower_band = calculate_bollinger_bands(df_stock)
        fig.add_trace(
            go.Scatter(x=df_stock.index, y=upper_band, mode='lines', name='Upper Bollinger Band', line=dict(color='green')),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=df_stock.index, y=rolling_mean, mode='lines', name='Middle Bollinger Band', line=dict(color='blue')),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(x=df_stock.index, y=lower_band, mode='lines', name='Lower Bollinger Band', line=dict(color='red')),
            secondary_y=True,
        )

    fig.update_layout(title=f'{selected_stock} Indicators')
    fig.update_yaxes(title_text="RSI", secondary_y=False)
    fig.update_yaxes(title_text="Bollinger Bands", secondary_y=True)

    return fig




if __name__ == '__main__':
    app.run_server(debug=True)
