import os
import pandas as pd
import matplotlib.pyplot as plt

def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbol, dates, colname = 'Adj Close'):
    """Read stock data for given symbol from CSV files."""
    df = pd.DataFrame(index=dates)
    try:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
    except Exception,e:
        print "Error in reading ",symbol,".csv"
        print e
        df_temp=pd.DataFrame()
    df_temp = df_temp.rename(columns={colname: symbol})
    
    df = df.join(df_temp)
    df=df.dropna()


    return df

def plot_data(df, symbol,title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    file_path=os.path.join("..","{}_result".format(symbol),'{}.png'.format(title))
    plt.savefig(file_path)
    plt.show()
