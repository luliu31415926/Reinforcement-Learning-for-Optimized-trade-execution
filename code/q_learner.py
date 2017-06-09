
import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.1, \
        gamma = 0.95, \
        rar = 0.5, \
        radr = 0.99,\
        verbose=False):

         # Initialize fields
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.verbose=verbose

        # Initialize state and action
        self.state = 0
        self.action = 0

        # Initialize Q with values ~ Uniform[-1, 1]
        self.Q = np.random.rand(num_states, num_actions) * 2 - 1
        if self.verbose: np.savetxt('initialQtable.txt', self.Q, delimiter=',')   

    def query_set_state(self, s):
        """
        @summary: selection action under state s according to the Q-table without updating the Q-table
        @param s: The current state
        @returns: The selected action according to the current Q-table 
        """
        self.state = s


        # Draw action from state s with the max q value.
        action = np.argmax(self.Q[s, :])
        if self.verbose: print "s =", s,"a =",action
        return action

    def query_and_update(self,s_prime,r):
        """
        @summary: Update the Q-table and return an action
        @param s_prime: The new state
        @param r: The reward for taking previous action
        @returns: The selected action
        """

        #Update Q[s,a]<- (1-alpha)*Q[s,a]+alpha(r+gamma*max(Q[s',:]))
        self.Q[self.state,self.action]=(1-self.alpha)*self.Q[self.state,self.action]+self.alpha*(r+self.gamma*np.max(self.Q[s_prime,:]))
        
        action = None

        #Choose random action with probability rar or draw action from the Q table 
        if rand.random()>self.rar:
            action=np.argmax(self.Q[s_prime,:])
        else:
            action=rand.randrange(self.num_actions)

        #decay random action rate: explore less as we are more sure of the best action to take 
        self.rar*=self.radr

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        self.state=s_prime
        self.action=action 
        return action
