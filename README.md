# Stock Market Data Analysis with Alpha Vantage

A streamlined Python script for fetching, transforming, and visualizing stock data from Alpha Vantage API, focusing on German/European markets.

## Project Overview

This project provides a simple, yet effective way to analyze stock market data using Python and the Alpha Vantage API. 
It focuses on practical implementation rather than complex architecture, making it ideal for quick market insights.

## Features

* Connect to Alpha Vantage API and fetch stock data
* Perform basic transformations on the data (daily returns, moving averages)
* Create meaningful visualizations for investment insights
* Focus on German/European markets

## Requirements

* Python 3.7+
* Required packages:
  * requests
  * pandas
  * numpy
  * matplotlib
  * seaborn
  
## Installation

1. Clone this repository:
```bash
git clone https://github.com/fdeligoez/stock-market-analysis-alphavantage.git
cd stock-market-analysis-alphavantage
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Get your Alpha Vantage API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)

4. Create a `config.py` file with your API key:
```python
API_KEY = "YOUR_API_KEY_HERE"
```

## Usage

```bash
python stock_analysis.py
```

Or use the Jupyter notebook:
```bash
jupyter notebook stock_analysis.ipynb
```

## Project Components

1. **API Connection**: Functions to fetch stock data from Alpha Vantage
2. **Data Transformation**: Cleaning and calculating metrics on the raw data
3. **Visualization**: Creating charts for price history and calculated metrics

## License

MIT
