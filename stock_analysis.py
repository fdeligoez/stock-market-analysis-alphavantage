"""
Stock Market Data Analysis with Alpha Vantage

This script fetches, transforms, and visualizes stock data using the Alpha Vantage API,
focusing on German/European markets.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import os
import sys

# Import configuration
try:
    from config import API_KEY, DEFAULT_SYMBOLS, DEFAULT_TIME_PERIOD, DEFAULT_MA_WINDOW
except ImportError:
    print("Error: config.py file not found or incomplete.")
    print("Please create a config.py file with API_KEY and other settings.")
    sys.exit(1)

# Set up plotting style
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


class StockAnalyzer:
    """Class for analyzing stock data from Alpha Vantage API"""

    def __init__(self, api_key=API_KEY):
        """Initialize with API key"""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.data = {}  # Dictionary to store fetched data

    def fetch_time_series(self, symbol, time_period=DEFAULT_TIME_PERIOD, outputsize="full"):
        """
        Fetch time series data from Alpha Vantage API

        Parameters:
        symbol (str): Stock symbol (e.g., "SAP.DEX" for SAP in German market)
        time_period (str): Time period for data - 'daily', 'weekly', or 'monthly'
        outputsize (str): 'compact' for last 100 datapoints, 'full' for all available data

        Returns:
        pandas.DataFrame: DataFrame with stock data
        """
        function_map = {
            "daily": "TIME_SERIES_DAILY",
            "weekly": "TIME_SERIES_WEEKLY",
            "monthly": "TIME_SERIES_MONTHLY"
        }

        # Map the function name based on time period
        function = function_map.get(time_period.lower(), "TIME_SERIES_DAILY")

        # Build parameters
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": outputsize
        }

        print(f"Fetching {time_period} data for {symbol}...")

        try:
            # Make API request
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            # Check for error messages
            if "Error Message" in data:
                print(f"API Error: {data['Error Message']}")
                return None

            # Extract the time series data
            time_series_key = f"Time Series ({time_period.capitalize()})"
            if time_period.lower() == "daily":
                time_series_key = "Time Series (Daily)"

            # Extract time series data
            time_series = data.get(time_series_key)
            if not time_series:
                print(f"No time series data found. Response: {data}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient="index")

            # Rename columns for consistency
            df.columns = [col.split('. ')[1] if '. ' in col else col for col in df.columns]
            df.columns = [col.lower() for col in df.columns]

            # Convert columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])

            # Add date as a column and sort by date
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            # Store data in the instance
            self.data[symbol] = df

            print(f"Successfully fetched data for {symbol}")
            return df

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
        except (KeyError, ValueError) as e:
            print(f"Error processing data: {e}")
            return None

    def calculate_metrics(self, symbol, ma_window=DEFAULT_MA_WINDOW):
        """
        Calculate various stock metrics

        Parameters:
        symbol (str): Stock symbol to analyze
        ma_window (int): Window size for moving averages

        Returns:
        pandas.DataFrame: DataFrame with original data plus calculated metrics
        """
        if symbol not in self.data:
            print(f"No data found for {symbol}. Fetch data first.")
            return None

        df = self.data[symbol].copy()

        # Calculate daily returns
        df['daily_return'] = df['close'].pct_change() * 100

        # Calculate moving averages
        df[f'ma_{ma_window}'] = df['close'].rolling(window=ma_window).mean()

        # Calculate MACD components
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Calculate volatility (standard deviation of returns over window)
        df['volatility'] = df['daily_return'].rolling(window=ma_window).std()

        # Update the stored data
        self.data[symbol] = df

        return df

    def plot_price_history(self, symbol, start_date=None, end_date=None, show_ma=True, ma_window=DEFAULT_MA_WINDOW):
        """
        Plot the price history with optional moving average

        Parameters:
        symbol (str): Stock symbol to plot
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        show_ma (bool): Whether to show moving average line
        ma_window (int): Window size for moving average

        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if symbol not in self.data:
            print(f"No data found for {symbol}. Fetch data first.")
            return None

        df = self.data[symbol].copy()

        # Filter by date range if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Check if we have the moving average column
        ma_col = f'ma_{ma_window}'
        if show_ma and ma_col not in df.columns:
            self.calculate_metrics(symbol, ma_window)
            df = self.data[symbol].copy()
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(df.index, df['close'], label='Close Price', linewidth=2)

        if show_ma and ma_col in df.columns:
            ax.plot(df.index, df[ma_col], label=f'{ma_window}-Day MA', linewidth=1.5, linestyle='--')

        # Customize the plot
        title = f"{symbol} Price History"
        if start_date or end_date:
            date_range = ""
            if start_date:
                date_range += f"from {start_date} "
            if end_date:
                date_range += f"to {end_date}"
            title += f" ({date_range})"

        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_returns_distribution(self, symbol):
        """
        Plot the distribution of daily returns

        Parameters:
        symbol (str): Stock symbol to analyze

        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if symbol not in self.data:
            print(f"No data found for {symbol}. Fetch data first.")
            return None

        df = self.data[symbol].copy()

        # Check if we have the daily returns column
        if 'daily_return' not in df.columns:
            self.calculate_metrics(symbol)
            df = self.data[symbol].copy()

        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Histogram
        sns.histplot(df['daily_return'].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f'{symbol} Daily Returns Distribution')
        axes[0].set_xlabel('Daily Return (%)')
        axes[0].set_ylabel('Frequency')

        # QQ Plot to check for normality
        from scipy import stats
        returns = df['daily_return'].dropna()
        stats.probplot(returns, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal Distribution Test)')

        plt.tight_layout()
        return fig

    def plot_volatility(self, symbol, ma_window=DEFAULT_MA_WINDOW):
        """
        Plot the stock's volatility over time

        Parameters:
        symbol (str): Stock symbol to analyze
        ma_window (int): Window size for volatility calculation

        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if symbol not in self.data:
            print(f"No data found for {symbol}. Fetch data first.")
            return None

        df = self.data[symbol].copy()

        # Check if we have the volatility column
        if 'volatility' not in df.columns:
            self.calculate_metrics(symbol, ma_window)
            df = self.data[symbol].copy()

        # Create the plot
        fig, ax = plt.subplots()
        ax.plot(df.index, df['volatility'], label=f'{ma_window}-Day Volatility', color='red')

        # Customize the plot
        ax.set_title(f"{symbol} Volatility ({ma_window}-Day Rolling Window)")
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility (%)')
        ax.legend()
        ax.grid(True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def plot_macd(self, symbol):
        """
        Plot MACD (Moving Average Convergence Divergence) indicator

        Parameters:
        symbol (str): Stock symbol to analyze

        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if symbol not in self.data:
            print(f"No data found for {symbol}. Fetch data first.")
            return None

        df = self.data[symbol].copy()

        # Check if we have MACD columns
        if 'macd' not in df.columns:
            self.calculate_metrics(symbol)
            df = self.data[symbol].copy()

        # Create the plot with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Price chart on top subplot
        ax1.plot(df.index, df['close'], label='Close Price')
        ax1.set_title(f"{symbol} Price and MACD")
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)

        # MACD chart on bottom subplot
        ax2.plot(df.index, df['macd'], label='MACD Line', color='blue')
        ax2.plot(df.index, df['signal'], label='Signal Line', color='red', linestyle='--')

        # Add MACD histogram
        hist_color = ['green' if val >= 0 else 'red' for val in (df['macd'] - df['signal'])]
        ax2.bar(df.index, df['macd'] - df['signal'], color=hist_color, alpha=0.5, label='Histogram')

        ax2.set_xlabel('Date')
        ax2.set_ylabel('MACD')
        ax2.legend()
        ax2.grid(True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def compare_stocks(self, symbols, metric='close', start_date=None, end_date=None):
        """
        Compare multiple stocks based on a specific metric

        Parameters:
        symbols (list): List of stock symbols to compare
        metric (str): Metric to compare ('close', 'daily_return', 'volatility', etc.)
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

        Returns:
        matplotlib.figure.Figure: The created figure
        """
        # Create empty DataFrame to store the comparison data
        comparison_df = pd.DataFrame()

        for symbol in symbols:
            if symbol not in self.data:
                print(f"No data found for {symbol}. Skipping.")
                continue

            df = self.data[symbol].copy()

            # Filter by date range if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            # Check if we have the requested metric
            if metric == 'close':
                comparison_df[symbol] = df['close']
            elif metric in df.columns:
                comparison_df[symbol] = df[metric]
            else:
                # Try to calculate the metric
                self.calculate_metrics(symbol)
                df = self.data[symbol].copy()
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]

                if metric in df.columns:
                    comparison_df[symbol] = df[metric]
                else:
                    print(f"Metric '{metric}' not available for {symbol}. Skipping.")
                    continue

        if comparison_df.empty:
            print("No data to compare.")
            return None

        # Normalize the data for better comparison
        if metric == 'close':
            normalized_df = comparison_df / comparison_df.iloc[0] * 100
        else:
            normalized_df = comparison_df

        # Create the plot
        fig, ax = plt.subplots()

        for column in normalized_df.columns:
            ax.plot(normalized_df.index, normalized_df[column], label=column, linewidth=2)

        # Customize the plot
        metric_name = metric.replace('_', ' ').title()
        title = f"Comparison of {metric_name}"
        if metric == 'close':
            title += " (Normalized to 100)"

        if start_date or end_date:
            date_range = ""
            if start_date:
                date_range += f"from {start_date} "
            if end_date:
                date_range += f"to {end_date}"
            title += f" ({date_range})"

        ax.set_title(title)
        ax.set_xlabel('Date')
        
        if metric == 'close' and len(normalized_df.columns) > 0:
            ax.set_ylabel('Normalized Price (Base=100)')
        else:
            ax.set_ylabel(metric_name)
            
        ax.legend()
        ax.grid(True)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def save_plots(self, symbol, output_dir="plots"):
        """
        Generate and save all plots for a given symbol

        Parameters:
        symbol (str): Stock symbol to analyze
        output_dir (str): Directory to save plots

        Returns:
        list: List of saved plot filenames
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        # Make sure we have calculated metrics
        if symbol in self.data:
            self.calculate_metrics(symbol)
        
        # Price history plot
        try:
            fig = self.plot_price_history(symbol)
            if fig:
                filename = f"{output_dir}/{symbol}_price_history.png"
                fig.savefig(filename)
                plt.close(fig)
                saved_files.append(filename)
        except Exception as e:
            print(f"Error saving price history plot: {e}")
        
        # Returns distribution plot
        try:
            fig = self.plot_returns_distribution(symbol)
            if fig:
                filename = f"{output_dir}/{symbol}_returns_distribution.png"
                fig.savefig(filename)
                plt.close(fig)
                saved_files.append(filename)
        except Exception as e:
            print(f"Error saving returns distribution plot: {e}")
        
        # Volatility plot
        try:
            fig = self.plot_volatility(symbol)
            if fig:
                filename = f"{output_dir}/{symbol}_volatility.png"
                fig.savefig(filename)
                plt.close(fig)
                saved_files.append(filename)
        except Exception as e:
            print(f"Error saving volatility plot: {e}")
        
        # MACD plot
        try:
            fig = self.plot_macd(symbol)
            if fig:
                filename = f"{output_dir}/{symbol}_macd.png"
                fig.savefig(filename)
                plt.close(fig)
                saved_files.append(filename)
        except Exception as e:
            print(f"Error saving MACD plot: {e}")
        
        return saved_files


def main():
    """Main function to run the analysis"""
    print("Stock Market Data Analysis with Alpha Vantage")
    print("-" * 50)
    
    # Check if API key is properly set
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Error: Please set your Alpha Vantage API key in config.py")
        return
    
    # Initialize the analyzer
    analyzer = StockAnalyzer()
    
    # Fetch data for default symbols
    for symbol in DEFAULT_SYMBOLS:
        analyzer.fetch_time_series(symbol, time_period=DEFAULT_TIME_PERIOD)
        # Sleep a bit to avoid hitting API rate limits
        time.sleep(1)
    
    # Generate plots for each symbol
    plots_dir = f"plots_{datetime.now().strftime('%Y%m%d')}"
    for symbol in DEFAULT_SYMBOLS:
        if symbol in analyzer.data:
            print(f"\nGenerating plots for {symbol}...")
            saved_files = analyzer.save_plots(symbol, output_dir=plots_dir)
            print(f"Saved {len(saved_files)} plots to {plots_dir}/")
    
    # Generate comparison plot
    symbols_with_data = [symbol for symbol in DEFAULT_SYMBOLS if symbol in analyzer.data]
    if len(symbols_with_data) > 1:
        print("\nGenerating comparison plot...")
        fig = analyzer.compare_stocks(symbols_with_data)
        if fig:
            comparison_file = f"{plots_dir}/comparison_normalized_price.png"
            fig.savefig(comparison_file)
            plt.close(fig)
            print(f"Saved comparison plot to {comparison_file}")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
