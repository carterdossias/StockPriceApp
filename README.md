# Stock Price Prediction App
The goal of this project is to use data pulled from various sources to train a ML model to predict closign price for a specified day.
There are many components to this project:

### Database
- There is a MySQL database running in my homelab storing all data in tables
- This database is currently managed by @NoahMalewicki
- Python scripts are use to automatically pull / process / import data into the DB

### Sentiment Analysis
- We have built python scripts that are designed to do sentiment analysis on the summaries of the news articles pulled for each day for a ticker
- The analysis will output a value between -1 and 1 and insert that value into a column in the database
- For each day a stock is traded, we pool all sentiment analysis for all articles written about that ticker

### ML Model
#### NOT CURRENTLY IMPLEMENTED

### Web App
- Finally, we added a web app that displays an interface to interact with the database


The raw stock data is currently pulled from Yahoo Finance using the yfinance library
The news data is currently pulled via API key from FINNUB