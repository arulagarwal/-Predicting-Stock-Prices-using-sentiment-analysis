from flask import Flask, request, render_template
import yfinance as yf
from getnews import get_articles
from GetPrediction import get_prediction
from datetime import datetime
import datetime

# instantiate the Flask app.
app = Flask(__name__)

# API Route for pulling Stock related news
@app.route("/news")
def get_news():
	symbol = request.args.get('symbol', default="AAPL")
	data = ""
	keywrds = {"AAPL":"Apple","MSFT":"Microsoft","TSLA":"Tesla"}
	name = keywrds[symbol]
	df = get_articles(name)
	data = ""
	for ind in df.index:
		date = "<div id = \"date\">"+df['publishedAt'][ind]+"</div>"
		source = "<div id = \"source\">"+df['name'][ind]+"</div>"
		title = "<div id = \"headline\">"+"<a href=\""+df['url'][ind]+"\">"+df['title'][ind]+"</a>"+"</div>"
		inp = date+source+title+"<br>"
		data = data+inp
	return data

# API Route for pulling the stock quote
@app.route("/quote")
def display_quote():
	# get a stock ticker symbol from the query string
	# default to AAPL
	symbol = request.args.get('symbol', default="AAPL")

	# pull the stock quote
	quote = yf.Ticker(symbol)

	#return the object via the HTTP Response
	return quote.info

# API route for pulling the stock history
@app.route("/history")
def display_history():
	#get the query string parameters
	symbol = request.args.get('symbol', default="AAPL")
	period = request.args.get('period', default="1mo")
	interval = request.args.get('interval', default="1d")

	#pull the quote
	quote = yf.Ticker(symbol)	
	#use the quote to pull the historical data from Yahoo finance
	hist = quote.history(period=period, interval=interval)
	y = datetime.datetime.now()
	dt = datetime.datetime(y.year, y.month, y.day)
	hist.loc[dt] = hist.iloc[0]
	hist.Close.loc[dt] = get_prediction(symbol)
	print(dt)
	print(hist.loc[dt])
	#convert the historical data to JSON
	data = hist.to_json()
	#return the JSON in the HTTP response
	return data

# This is the / route, or the main landing page route.
@app.route("/")
def home():
	# we will use Flask's render_template method to render a website template.
    return render_template("homepage.html")

# run the flask app.
if __name__ == "__main__":
	app.run(debug=True)