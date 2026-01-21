import pandas as pd
import requests


def create_local_sp500():
    url = 'https://www.slickcharts.com/sp500'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    response = requests.get(url, headers=headers)
    # Extract the first table found on the page
    df = pd.read_html(response.text)[0]

    # Standardize tickers for yfinance (dots to dashes)
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)

    # Save specifically as sp500.csv
    df[['Symbol', 'Company']].to_csv("sp500.csv", index=False)
    print("Success: Wikipedia-free database 'sp500.csv' created.")


create_local_sp500()