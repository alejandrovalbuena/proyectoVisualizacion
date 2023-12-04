import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import yfinance as yf
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


app = dash.Dash(__name__)

top_50_sp500 = [
    {'label': 'Apple Inc', 'value': 'AAPL'},
    {'label': 'Broadcom Inc', 'value': 'AVGO'},
    {'label': 'Sysco Corporation', 'value': 'SYY'},
    {'label': 'T-Mobile US', 'value': 'TMUS'},
    {'label': 'T. Rowe Price', 'value': 'TROW'},
    {'label': 'Take-Two Interactive', 'value': 'TTWO'},
    {'label': 'Tapestry, Inc.', 'value': 'TPR'},
    {'label': 'Targa Resources', 'value': 'TRGP'},
    {'label': 'Target Corporation', 'value': 'TGT'},
    {'label': 'TE Connectivity', 'value': 'TEL'},
    {'label': 'Tesla, Inc.', 'value': 'TSLA'},
    {'label': 'Texas Instruments', 'value': 'TXN'},
    {'label': 'Textron', 'value': 'TXT'},
    {'label': 'Thermo Fisher Scientific', 'value': 'TMO'},
    {'label': 'TJX Companies', 'value': 'TJX'},
    {'label': 'Tractor Supply Company', 'value': 'TSCO'},
    {'label': 'Trane Technologies', 'value': 'TT'},
    {'label': 'TransDigm Group', 'value': 'TDG'},
    {'label': 'Travelers Companies', 'value': 'TRV'},
    {'label': 'Trimble Inc.', 'value': 'TRMB'},
    {'label': 'Truist Financial', 'value': 'TFC'},
    {'label': 'Tyler Technologies', 'value': 'TYL'},
    {'label': 'Tyson Foods', 'value': 'TSN'},
    {'label': 'U.S. Bancorp', 'value': 'USB'},
    {'label': 'UDR, Inc.', 'value': 'UDR'}
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

def fetch_hourly_stock_data(symbol, start_date, end_date):
    # Fetches hourly stock data within a specified date range
    stock = yf.Ticker(symbol)
    df = stock.history(interval="1h", start=start_date, end=end_date)
    return df

def perform_hourly_linear_regression(dataframe, hours_ahead=24):
    # Ensure the dataframe is sorted by date
    dataframe = dataframe.sort_index()
    hourly_data = dataframe
    
    hourly_data['NumericIndex'] = range(len(hourly_data))
    X = hourly_data[['NumericIndex']] 
    y = hourly_data['Close']


    # Fit the regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict future values
    last_numeric_index = hourly_data['NumericIndex'].iloc[-1]
    future_indices = np.array(range(last_numeric_index + 1, last_numeric_index + hours_ahead + 1)).reshape(-1, 1)
    future_preds = model.predict(future_indices)
    
    # Calculate the RMSE for confidence intervals
    y_pred = model.predict(X)
    rmse = sqrt(mean_squared_error(y, y_pred))
    
    # Create confidence intervals
    lower_bound = future_preds - 1.96 * rmse
    upper_bound = future_preds + 1.96 * rmse
    
    return hourly_data.index, hourly_data['Close'], future_preds, lower_bound, upper_bound, model



app.layout = html.Div([
    html.H1('S&P 500 Stock App'),

    dcc.Dropdown(
        id='stock-selector',
        options=top_50_sp500,
        value='AAPL'  # Default value
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
        value='1y'  # Default value
    ),

    html.Button('Select All Indicators', id='select-all-button'),

    dcc.Checklist(
        id='ma-options-selector',
        options=[
            {'label': 'Show 200-session Moving Average', 'value': 'MA200'},
            {'label': 'Show 100-session Moving Average', 'value': 'MA100'},
            {'label': 'Show 50-session Moving Average', 'value': 'MA50'}
        ],
        value=[]  # No default value, user must select
    ),

    dcc.Graph(id='stock-price-graph'),

    dcc.Checklist(
        id='indicator-selector',
        options=[
            {'label': 'Show RSI', 'value': 'RSI'},
            {'label': 'Show Bollinger Bands', 'value': 'BOLL'}
        ],
        value=[]  # No default value, user must select
    ),

    dcc.Graph(id='indicator-graph'),
    dcc.Graph(id='linear-regression-prediction-graph')
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
    Output('linear-regression-prediction-graph', 'figure'),
    [Input('stock-selector', 'value'), Input('time-range-selector', 'value')]
)
def update_hourly_linear_regression_graph(selected_stock, selected_time_range):
    # Define the start and end date for fetching hourly data
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(weeks=4)  # Fetching 4 weeks of data for example

    df_stock = fetch_hourly_stock_data(selected_stock, start_date, end_date)
    if df_stock.empty:
        return go.Figure()

    # Calculate polynomial regression
    x = np.array(range(len(df_stock.index))).reshape(-1, 1)
    y = df_stock['Close'].values
    poly = PolynomialFeatures(degree=3)
    x_poly = poly.fit_transform(x)
    poly.fit(x_poly, y)
    lin2 = LinearRegression()
    lin2.fit(x_poly, y)

    # Create future dates
    future_dates = pd.date_range(start=df_stock.index[-1], periods=24, freq='H')  # Predicting the next 24 hours for example

    # Apply polynomial regression function to future dates
    future_preds = lin2.predict(poly.fit_transform(np.array(range(len(df_stock.index), len(df_stock.index) + len(future_dates))).reshape(-1, 1)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_stock.index, y=df_stock['Close'], mode='lines', name='Historical Prices'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Polynomial Regression Predictions', line=dict(color='orange')))

    fig.update_layout(title='Hourly Stock Price Prediction with Polynomial Regression')
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)