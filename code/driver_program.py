import pandas as pd
import datetime as dt
import util as ut
import policy_learner as pl


def test_code(verbose = True):

    # instantiate the strategy learner
    learner = pl.PolicyLearner(verbose = verbose)

    # set parameters for training the learner
    sym = "GOOG"
    stdate =dt.datetime(2015,1,1)
    enddate =dt.datetime(2016,1,1) 

    # train the learner
    learner.add_evidence(symbol = sym, sd = stdate, \
        ed = enddate) 

    # set parameters for testing
    sym = "GOOG"
    stdate =dt.datetime(2016,1,1)
    enddate =dt.datetime(2016,4,1)


    # test the learner
    learner.test_policy(symbol = sym, sd = stdate, ed = enddate)


if __name__=="__main__":
    test_code(verbose = False)
