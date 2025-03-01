"""
Simple example of using the StockAnalyzer class
"""

import matplotlib.pyplot as plt
from datetime import datetime
import os
from stock_analysis import StockAnalyzer

# Set your API key here (or use config.py)
API_KEY = "YOUR_API_KEY_HERE"

def main():
    """Example script showing how to use the StockAnalyzer"""
    
    print("Stock Market Analysis Example")
    print("-" * 40)
    
    # Initialize the analyzer
    analyzer = StockAnalyzer(api_key=API_KEY)
    
    # Symbols to analyze (German stocks)
    symbols = ["SAP.DEX", "ALV.DEX"]
    
    # Fetch data for each symbol
    for symbol in symbols:
        analyzer.fetch_time_series(symbol)
    
    # Generate plots directory
    today = datetime.now().strftime('%Y%m%d')
    plots_dir = f"example_plots_{today}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate analysis for each stock
    for symbol in symbols:
        if symbol in analyzer.data:
            # Calculate metrics
            analyzer.calculate_metrics(symbol)
            
            # Create and save plots
            print(f"\nGenerating plots for {symbol}...")
            
            # Price history plot
            fig = analyzer.plot_price_history(symbol, start_date="2023-01-01")
            fig.savefig(f"{plots_dir}/{symbol}_price.png")
            plt.close(fig)
            
            # MACD plot
            fig = analyzer.plot_macd(symbol)
            fig.savefig(f"{plots_dir}/{symbol}_macd.png")
            plt.close(fig)
            
            print(f"Plots saved to {plots_dir}/")
    
    # Create comparison plot
    if len(symbols) > 1:
        fig = analyzer.compare_stocks(symbols, start_date="2023-01-01")
        fig.savefig(f"{plots_dir}/comparison.png")
        plt.close(fig)
        print(f"\nComparison plot saved to {plots_dir}/comparison.png")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
