
""" 
Rules: 
limit to trade 100 shares per day 
Actions: 0 = BUY, 1 = SELL, 2 = NOTHING
Positions: 0 = CASH, 1 = SHORT, 2 = LONG
states= (Bollinger Band Value) + (volatility)*10+ (volume)*100+Position*1000
2999 different states   
start trading with long 100 shares 
starting portfolio value =price of 100 shares 
assume no trasaction fees and trading does not affect stock price 
"""
import os
import time
import datetime as dt
import q_learner as ql
import pandas as pd
import util as ut
import numpy as np

class PolicyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        #initiate a Q learner 
        self.learner=ql.QLearner(num_states=3000,num_actions=3,rar=0.5, radr=0.99)



    """
    Train Learner Functions 
    """
    def add_evidence(self,symbol,sd,ed): 
    #Feed data to the QLearner, and train it for trading

        self.symbol=symbol
        self.sd=sd
        self.ed=ed

        #calculate features 
        df_features=self.calc_features()
        #print "training ..."
        #print df_features
        if np.sum(df_features.values)==0:
            print 'NO FEATURES!'
        else: 
            #calculate discretize thresholds
            self.thresholds=self.disc_thresh(df_features)
            #train QLearner 
            self.train_learner(df_features)



    def train_learner(self,df_features):

        num_of_trials=80
        #initialize holdings 
        df_original_holdings=self.build_holdings(df_features)

        #dataframe to store result for each trial 
        df_training=pd.DataFrame(index=range(0,num_of_trials),columns=['Cummulative Return'])
        #Training loop 
        for i in range (0,num_of_trials):
            #reset df_holdings
            df_holdings=df_original_holdings.copy()
            #start with long 100 shares position
            cur_pos=2 
            date=0

            state = self.discretize(df_features.ix[date,'BB'], \
                                    df_features.ix[date,'VOLAT'], \
                                    df_features.ix[date,'VOLUME']) + cur_pos * 1000

            action = self.learner.query_set_state(state)
            
            
            for date in range(0,df_features.shape[0]):
                

                #if (i==num_of_trials-1):
                    #print date, state, action , df_holdings.ix[date,'Portfolio Value']
                #apply selected action and update df_holdings and cur_pos
                df_holdings,cur_pos=self.apply_action(date,df_holdings,action)
                #calculate the reward - change in portfolio value from previous day, after applying the selected action
                reward=df_holdings.ix[date,'Portfolio Value']-df_holdings.ix[date-1,'Portfolio Value']

                #update state by discretizing 
                state=self.discretize(df_features.ix[date,'BB'],df_features.ix[date,'VOLAT'],df_features.ix[date,'VOLUME'])+cur_pos*1000
                #select action to apply for tomorrow, also update Q-table 
                action=self.learner.query_and_update(state,reward)

        
                if self.verbose: print state,reward, df_holdings.ix[date,'Portfolio Value']
            #by the end of the one trial, calculate cummulative portfolio return 
            cum_port_return=df_holdings.ix[-1,'Portfolio Value']/df_holdings.ix[0,'Portfolio Value']-1
            print i,'Cummulative Return: ',cum_port_return
            df_training.ix[i,'Cummulative Return']=cum_port_return

        if self.verbose: print df_training
        #store training performance states to local file 
        file_path=os.path.join("..","{}_result".format(self.symbol), 'Training Performance_{}.csv'.format(self.symbol))
        df_training.to_csv(file_path, sep='\t')
        #plot training performance statistics 
        ut.plot_data(df_training, symbol=self.symbol, title="Training Performance", xlabel="# of trials", ylabel="Cummulative Portfolio Return")
        
        #apply learnt policy to the training dataset, to see training result 
        self.test_policy(symbol = self.symbol, sd = self.sd, ed = self.ed,is_training=True)

    

    """
    Test Learner Functions
    """
    #use feature information for each day, 
    #predict trade strategy for the next day according to the learnt q table 
    def trade_generator(self, df_features):
        #Actions: 0 = BUY, 1 = SELL, 2 = NOTHING
        BUY = 0
        SELL = 1
        NOTHING = 2
        #Positions: 0 = CASH, 1 = SHORT, 2 = LONG
        CASH = 0
        SHORT = 1
        LONG = 2

        #starting trading with long 100 shares 
        cur_pos = LONG
        holdings = 100
        #obtain the state for the first day and action to take for the next day 
        date=0
        state = self.discretize(df_features.ix[date,'BB'], \
                                df_features.ix[date,'VOLAT'], \
                                df_features.ix[date,'VOLUME']) + cur_pos * 1000

        action = self.learner.query_set_state(state)
        #loop through every trading day to determine the trade strategy for the next day  
        for date in range(0, df_features.shape[0]):
          
            if action == BUY and cur_pos != LONG:
                trade = 100
                holdings += 100
            elif action == SELL and cur_pos != SHORT:
                trade = -100
                holdings -= 100
            else:
                trade = 0

            yield trade   

            if holdings == 100:
                cur_pos = LONG
            elif holdings == -100:
                cur_pos = SHORT
            else:
                cur_pos = CASH
        
            #obtain today's end of the day state, and obtain action to take for the next day 
            state = self.discretize(df_features.ix[date,'BB'], \
                                df_features.ix[date,'VOLAT'], \
                                df_features.ix[date,'VOLUME']) + cur_pos * 1000
            action = self.learner.query_set_state(state) 

    def test_policy(self,symbol,sd,ed,is_training=False):
    #test learnt policy on data provided 
        self.symbol=symbol
        self.sd=sd
        self.ed=ed 

        #set different file name to distinguish from training dataset and testing dataset 
        if (is_training):
            file_name=self.symbol+"_train"
        else:
            file_name=self.symbol

        #calculate the features
        df_features=self.calc_features()
        #check of features are properly created 
        if np.sum(df_features.values)==0:
            print 'NO FEATURES!'

        #create a dataframe to store actions for review purpose 
        df_trades=pd.DataFrame(index=df_features.index, columns=['Trades'])

        #build a holding profile to start with, make a copy for making transactions 
        df_original_holdings=self.build_holdings(df_features)
        df_holdings=df_original_holdings.copy()
        date=0

        for trade in self.trade_generator(df_features):
            #apply trade to holdings
            df_holdings.ix[date:,self.symbol]+=trade
            df_holdings.ix[date:,'Cash']-=trade*df_holdings.ix[date,'Stock Price']         
            df_holdings.ix[date:,'Portfolio Value']=df_holdings.ix[date,'Cash']+(df_holdings.ix[date,self.symbol]*df_holdings.ix[date,'Stock Price'])
            #store trade in df_trades dataframe 
            df_trades.ix[date,'Trades']=trade
            date+=1

        

        #save the calculated action plan to local file 
        file_path=os.path.join("..","{}_result".format(self.symbol), 'CalculatedTrades_{}.csv'.format(file_name))
        df_trades.to_csv(file_path,sep='\t')


        #save result to local file 
        file_path=os.path.join("..","{}_result".format(self.symbol), 'BackTestResult_{}.csv'.format(file_name))
        df_holdings.to_csv(file_path,sep='\t')

        #plot value of 100 shares of stock and value of the portfolio for comparison 
        df_testing=pd.DataFrame(index=df_holdings.index,columns=['100 x Stock Price','Portfolio Value'])
        df_testing['100 x Stock Price']=100*df_features['Price']
        df_testing['Portfolio Value']=df_holdings['Portfolio Value']
        ut.plot_data(df_testing[20:], symbol=self.symbol,title=file_name+" Stock prices & Portfolio Value", xlabel="Date", ylabel="Value")

        cum_port_return=df_holdings.ix[-1,'Portfolio Value']/df_holdings.ix[20,'Portfolio Value']-1
        print 'Cummulative Return: for {} is {}%'.format(file_name,cum_port_return*100)
        



    """
    Discretize Functions
    """
    def discretize(self,bb,volat,volume):
        disc_val=0
        for i in range(0,self.disc_steps):
            if bb<=self.thresholds[i,0]:
                disc_val+=i
                break
        for i in range(0,self.disc_steps):
            if volat<=self.thresholds[i,1]:
                disc_val+=i*10
                break
        for i in range(0,self.disc_steps):
            if volume<=self.thresholds[i,2]:
                disc_val+=i*100
                break
        return disc_val

    def disc_thresh(self,df_features):
        thresholds=np.zeros((10,3))
        #data.sort()
        df_bb=df_features['BB'].copy()
        df_bb.sort_values(inplace=True)
        df_volat=df_features['VOLAT'].copy()
        df_volat.sort_values(inplace=True)
        df_volume=df_features['VOLUME'].copy()
        df_volume.sort_values(inplace=True)


        #define number of busckets
        self.disc_steps=steps=10
        stepsize=df_bb.size/steps

       
        for i in range (0,steps):
            #bb threshold
            thresholds[i,0]=df_bb[(i+1)*stepsize-1]
            #volatility threshold 
            thresholds[i,1]=df_volat[(i+1)*stepsize-1]
            #volumes threshold
            thresholds[i,2]=df_volume[(i+1)*stepsize-1]
        return thresholds


    """
    build features dataframe
    """
    def calc_stock_daily_ret(self,df_prices):
        normed=df_prices /df_prices.ix[0]
        stock_daily_ret=(normed/normed.shift(1))-1
        stock_daily_ret=stock_daily_ret[1:]
        return stock_daily_ret


    def calc_features(self):
        #window parameter for calculating rolling mean and standard devication
        window=20
        #get stock prices for symbol 
        df_prices=ut.get_data(symbol=self.symbol,dates=pd.date_range(self.sd,self.ed),colname='Adj Close')
        #print df_prices
        #get stock trading volume for symbol 
        df_volumes=ut.get_data(symbol=self.symbol,dates=pd.date_range(self.sd,self.ed),colname='Volume')

        #calculate dataframes for features:

        #stock daily returns 
        df_stock_daily_ret=self.calc_stock_daily_ret(df_prices)
        #simple moving average of staock prices
        df_sma=df_prices.rolling(window=window).mean()
        #standard deviation of stock prices 
        df_std=df_prices.rolling(window=window).std()
        #standard deviation of stock daily returns 
        df_stock_daily_ret_std=df_stock_daily_ret.rolling(window=window).std()
        columns=['BB','VOLAT','VOLUME','Price']
        index=df_sma.index
        df_features=pd.DataFrame(index=index,columns=columns)
        df_features['Price']=df_prices[self.symbol]
        #calculate features for each day 
        for t in range(window-1, df_prices.shape[0]-1):
            df_features.ix[t,'BB']=(df_prices.ix[t,self.symbol]-df_sma.ix[t,self.symbol])/(2*df_std.ix[t,self.symbol])
            df_features.ix[t,'VOLAT']=df_stock_daily_ret_std.ix[t,self.symbol]
            df_features.ix[t,'VOLUME']=df_volumes.ix[t,self.symbol]


        #print df_features.fillna(0)
        return df_features.fillna(0)


    """
    Utility Functions 
    """
    def build_holdings(self, df_features):
        #start the portfolio with holding 100 shares of the stock 
        columns=[self.symbol,'Stock Price','Cash','Portfolio Value']
        df_holdings=pd.DataFrame(index=df_features.index,columns=columns)
        df_holdings.ix[:,self.symbol]=100
        df_holdings.ix[:,'Stock Price']=df_features.ix[:,'Price']
        df_holdings.ix[:,'Cash']=0            
        df_holdings.ix[:,'Portfolio Value']=df_holdings.ix[0,self.symbol]*df_holdings.ix[0,'Stock Price']
        return df_holdings


    def calc_cur_pos(self,df_holdings,date):
        #position: 0=CASH, 1=SHORT, 2=LONG
        if df_holdings.ix[date,self.symbol]==0 :
            cur_pos=0 #cash position 
        elif df_holdings.ix[date,self.symbol]<0:                 
            cur_pos=1 #short position
        else: 
            cur_pos=2 #long position
        return cur_pos 


    def apply_action(self,date,df_holdings,action):
        #ACTIONS
        BUY=0
        SELL=1
        NOTHING=2 
        #POSITIONS 
        CASH=0
        SHORT=1
        LONG=2 
        #calculate current position 
        cur_pos=self.calc_cur_pos(df_holdings,date)
        #apply action if condition allows 
        if action==BUY and cur_pos!=LONG:
            #add 100 shares and deduct purchase cost from cash, extend the transaction into the future 
            df_holdings.ix[date:,self.symbol]+=100
            df_holdings.ix[date:,'Cash']=df_holdings.ix[date,'Cash']-df_holdings.ix[date,'Stock Price']*100
        elif action==SELL and cur_pos!=SHORT:
            #subtract 100 shares and add values to cash, extend the transaction into the future 
            df_holdings.ix[date:,self.symbol]-=100
            df_holdings.ix[date:,'Cash']=df_holdings.ix[date,'Cash']+df_holdings.ix[date,'Stock Price']*100   

        cur_pos=self.calc_cur_pos(df_holdings,date)
        #update portolio value and extend into the future 
        df_holdings.ix[date:,'Portfolio Value']=df_holdings.ix[date,'Cash']+df_holdings.ix[date,self.symbol]*df_holdings.ix[date,'Stock Price']
        return df_holdings, cur_pos

