'''
TODO
    - implement SIS
    - check if #recovered in plots makes sense
'''

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class ContagionModel():
    def __init__(self, graph, states, model, duration_infectious, infection_rate):
        self.graph = graph.copy()  # networkx graph, graph_u
        self.model = model  # string, SIR, SIRD, SIS
        self.duration = duration_infectious  # duration of infected after which it is recovered
        self.inf_rate = infection_rate  # probability of S -> I if S a neighbor of I
        self.states = states  # {node:state}
        self.history = [self.states.copy()]  # list of node states for each time "run" is called 
        self.track_inf = {k:1 for k,v in self.history[0].items() if v == 'I'}  # {infected_node:duration_of_its_infection}
        self.terminate = False
        
        
    def run(self):
        if self.model == 'SIR':
            self.SIR()
        elif self.model == 'SIS':
            self.SIS()
        else:
            raise ValueError(f'The contagion model {self.model} is not supported.')
            
            
    def SIR(self):
        ### Recovery I -> R ###
        recovered = {k:'R' for k,v in self.track_inf.items() if v >= self.duration}
        # update states
        self.states.update(recovered.copy())
        # update track_inf
        for rec in recovered:
            self.track_inf.pop(rec, None)
        self.track_inf = {k:v+1 for k,v in self.track_inf.items()}  # for next timestamp

        ### check for termination ###
        if len(self.track_inf) == 0:
            # update history
            self.history.append(self.states.copy())
            self.terminate = True
            return 

        ### Infection S -> I ###
        infected = {}
        for inf in self.track_inf:
            for neigh in self.graph.neighbors(inf):
                if self.states[neigh] == 'S' and np.random.random(1)[0] <= self.inf_rate:
                    infected[neigh] = 'I'
        # update states
        self.states.update(infected.copy())
        # update track_inf
        self.track_inf.update({k:1 for k in infected})

        # update history
        self.history.append(self.states.copy())

    def SIS(self):
        ### recovered I -> S ###
        recovered = {k:'S' for k,v in self.track_inf.items() if v >= self.duration}
        # update states
        self.states.update(recovered.copy())
        # update track_inf
        for rec in recovered:
            self.track_inf.pop(rec, None)
        self.track_inf = {k:v+1 for k,v in self.track_inf.items()}  # for next timestamp

        ### check for termination ###
        if len(self.track_inf) == 0:
            # update history
            self.history.append(self.states.copy())
            self.terminate = True
            return 

        ### Infection S -> I ###
        infected = {}
        for inf in self.track_inf:
            for neigh in self.graph.neighbors(inf):
                if self.states[neigh] == 'S' and np.random.random(1)[0] <= self.inf_rate:
                    infected[neigh] = 'I'
        # update states
        self.states.update(infected.copy())
        # update track_inf
        self.track_inf.update({k:1 for k in infected})

        # update history
        self.history.append(self.states.copy())
   
    def get_settings(self):
        return dict(graph = self.graph, model = self.model, duration = self.duration,
                     inf_rate = self.inf_rate)

    def get_states(self):
        return self.states

    def get_history(self):
        return self.history

    def terminate(self):
        return self.terminate

    def set_graph(self, g):
        self.graph = g
    
    def plot_history(self):
        if self.model == 'SIR':
            x = range(len(self.history))
            size = self.graph.number_of_nodes()
            num_inf = []
            num_sus = []
            num_rec = []
            for idx in x:
                counts = dict(Counter(self.history[idx].values()))
                num_inf.append(counts.pop('I', 0)/size)
                num_sus.append(counts.pop('S', 0)/size)
                num_rec.append(counts.pop('R', 0)/size)
            plt.figure(figsize=(10,5))
            plt.plot(x, num_inf, label = 'infected')
            # plt.plot(x, num_sus, label = 'susceptible')
            plt.plot(x, num_rec, label = 'recoverd')
            plt.xlabel('timestamp')
            plt.title(self.model)
            plt.legend()
            plt.show()
            
        elif self.model == 'SIS':
            x = range(len(self.history))
            size = self.graph.number_of_nodes()
            num_inf = []
            num_sus = []
            for idx in x:
                counts = dict(Counter(self.history[idx].values()))
                num_inf.append(counts.pop('I', 0)/size)
                num_sus.append(counts.pop('S', 0)/size)
            plt.figure(figsize=(10,5))
            plt.plot(x, num_inf, label = 'infected')
            plt.plot(x, num_sus, label = 'susceptible')
            plt.xlabel('timestamp')
            plt.title(self.model)
            plt.legend()
            plt.show()
            
        else:
            raise ValueError(f'The model {self.model} is not supported.')
    