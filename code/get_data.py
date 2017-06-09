from datetime import datetime
import pandas_datareader.data as web

# Specify Date Range
start = datetime(2000, 1, 1)
end = datetime.today()

# Specify symbol
symbols = ['AAPL','GOOG','IBM','SPY','CELG']
for symbol in symbols:
	from_yahoo = web.DataReader("%s" % symbol, 'yahoo', start, end)
	from_yahoo.to_csv('%s.csv' % symbol)