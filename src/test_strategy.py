'''
TODO: 
    - epsiolon with selected memory not implemented -- needs info from mitigation
'''

import numpy as np
from collections import Counter

class TestStrategy():
    def __init__(self, method, graph_unknown, spread_model, graph_known, test_budget, 
                 trace_acc, history_enable=True, **parameters):
        # immutable
        self.method = method  # the testing strategy name
        self.graph_u = graph_unknown.copy()  # the original graph that is unknown and to be learned via testing
        self.spread = spread_model  # the spread model that is operating on the original graph, contains node state
        self.history_flag = history_enable
        self.budget = test_budget  # in terms of number of nodes that can be tested in each run of test
        self.trace_acc = trace_acc  # 0<  <1. The proportion of neighbors a node reports.
        
        # mutable
        self.graph_k = graph_known.copy()  # the graph that we are going know and update via the test strategy
        self.history = []  # [([nodes_tested],[their_states], g_k info: {num_nodes: , num_edges: , {degree: count}})]
        self.current_results = ()  # ([nodes_tested],[their_states], g_k info: {num_nodes: , num_edges: , {degree: count}})) at current time
        self.params = parameters  # dict of test-specific parameters
        # {visited:[]} for random_wMemory
        # {epsilon: , rec_pos:[], decay_factor:} for epsilon greedy, rec_pos is positively tested in last test,
        # decay factor is for epsilon update --> new_ep = ep * decay_factor
        
    def run(self):
        if self.method == 'random':
            self.random()
        elif self.method == 'random_with_memory':
            self.random_wMemory()
        elif self.method == 'degree_with_memory':
            self.degree_wMemory()
        elif self.method == 'epsilon_greedy':
            self.epsilon_greedy()
        elif self.method == 'epsilon_memory':
            self.epsilon_memory()
        elif self.method == 'epsilon_degree':
            self.epsilon_degree()
        else:
            raise ValueError(f'The test strategy {self.method} is not supported.')
            
    def update(self, nodes_tested):
        test_results = np.asarray([self.spread.get_states()[k] for k in nodes_tested])
        # pickle neighbors randomly using self.trace_acc
        edges_collected = np.asarray([(node,neigh) for node in nodes_tested for \
                                      neigh in np.random.choice(list(self.graph_u.neighbors(node)), \
                                                                int(self.graph_u.degree(node)*self.trace_acc), \
                                                                replace=False)])
        self.graph_k.add_edges_from(edges_collected)
        self.current_results = (nodes_tested, test_results, self.get_graph_info(self.graph_k))
        if self.history_flag:
            self.history.append(self.current_results)
            
    def get_graph_info(self, graph):
        return dict(num_edges = graph.number_of_edges(),
                   deg_dist = dict(Counter(dict(graph.degree()).values())))
    
    def random(self):
        ''' Choose nodes at random. '''
        nodes_tested = np.random.choice(self.graph_k, self.budget, replace = False)
        self.update(nodes_tested)
            
    def random_wMemory(self):
        ''' Choose nodes at random but not the nodes visited in t-1. '''
        if not isinstance(self.params['visited'], np.ndarray):
            self.params['visited'] = np.asarray(self.params['visited'])
        visited = self.params['visited']
        candidates = list(set(list(self.graph_u.nodes())) ^ set(visited))
        if len(candidates) <= self.budget:
            nodes_tested = np.asarray(candidates)
        else:
            nodes_tested = np.random.choice(candidates, self.budget, replace = False)
        self.update(nodes_tested)
        self.params['visited'] = nodes_tested

    def degree_wMemory(self):
        ''' Choose nodes based on their degree excluding those visited at t-1. '''
        if not isinstance(self.params['visited'], np.ndarray):
            self.params['visited'] = np.asarray(self.params['visited'])
        visited = self.params['visited']
        candidates = list(set(list(self.graph_u.nodes())) ^ set(visited))
        sort_cand = sorted(candidates, key=lambda x: self.graph_k.degree(x), reverse=True)  # descending
        nodes_tested = np.asarray(sort_cand[:self.budget])
        self.update(nodes_tested)
        self.params['visited'] = nodes_tested
        
    def epsilon_greedy(self):
        # optimal action = find positive tests
        epsilon = self.params['epsilon']
        rec_pos = self.params['rec_pos']
        probs = np.random.random_sample((int(self.budget), ))
        p_eps = [i for i in probs if i < epsilon]
        # add optimal actions (random neighbor of rec_pos subset)
        if len(probs)-len(p_eps) > len(rec_pos):
            pivots = rec_pos
        else:
            pivots = np.random.choice(rec_pos, len(probs)-len(p_eps), replace=False)
        candidates = np.asarray(list(set([np.random.choice(list(self.graph_u.neighbors(node))) for node in pivots])))
        # add random actions (randomly choose a node to test)
        nodes_tested = np.random.choice(self.graph_u.nodes(), len(probs)-len(candidates), replace = False)
        nodes_tested = np.append(nodes_tested, candidates)
        self.update(nodes_tested)
        # self.params['epsilon'] = epsilon * (1/self.params['decay_factor'])
        # self.params['epsilon'] = epsilon ** self.params['decay_factor']
        self.params['epsilon'] = max(epsilon - epsilon/(self.params['decay_factor']+0.001), 0)
        test_result = self.current_results[1]
        self.params['rec_pos'] = np.asarray([nodes_tested[i] for i in range(len(nodes_tested)) if test_result[i]=='I'])

    def epsilon_memory(self): 
        if not isinstance(self.params['visited'], np.ndarray):
            self.params['visited'] = np.asarray(self.params['visited'])
        visited = self.params['visited']
        # optimal action = find positive tests
        epsilon = self.params['epsilon']
        rec_pos = self.params['rec_pos']
        probs = np.random.random_sample((int(self.budget), ))
        p_eps = [i for i in probs if i < epsilon]
        # add optimal actions (random neighbor of rec_pos subset)
        if len(probs)-len(p_eps) > len(rec_pos):
            pivots = rec_pos
        else:
            pivots = np.random.choice(rec_pos, len(probs)-len(p_eps), replace=False)
        candidates = np.asarray(list(set([np.random.choice(list(self.graph_u.neighbors(node))) for node in pivots])))
        # add random actions (randomly choose a node to test) -- ad it with memory
        not_visited = list(set(list(self.graph_u.nodes())) ^ set(visited))
        if len(not_visited) <= len(probs)-len(candidates):
            nodes_tested = np.asarray(not_visited)
        else:
            nodes_tested = np.random.choice(not_visited, len(probs)-len(candidates), replace = False)
        nodes_tested = np.append(nodes_tested, candidates)
        self.update(nodes_tested)
        # self.params['epsilon'] = epsilon * (1/self.params['decay_factor'])
        # self.params['epsilon'] = epsilon ** self.params['decay_factor']
        self.params['epsilon'] = max(epsilon - epsilon/(self.params['decay_factor']+0.001), 0)
        test_result = self.current_results[1]
        self.params['rec_pos'] = np.asarray([nodes_tested[i] for i in range(len(nodes_tested)) if test_result[i]=='I'])
        self.params['visited'] = nodes_tested
 
    def epsilon_degree(self): 
        if not isinstance(self.params['visited'], np.ndarray):
            self.params['visited'] = np.asarray(self.params['visited'])
        visited = self.params['visited']
        # optimal action = find positive tests
        epsilon = self.params['epsilon']
        rec_pos = self.params['rec_pos']
        rec_pos = sorted(rec_pos, key=lambda x: self.graph_k.degree(x), reverse= True)  # descending order
        probs = np.random.random_sample((int(self.budget), ))
        p_eps = [i for i in probs if i < epsilon]
        # add optimal actions (random neighbor of rec_pos subset)
        pivots = rec_pos[:len(probs)-len(p_eps)]
        candidates = np.asarray(list(set([np.random.choice(list(self.graph_u.neighbors(node))) for node in pivots])))
        # add random actions (choose a node to test by known degree) -- with memory
        not_visited = list(set(list(self.graph_u.nodes())) ^ set(visited))
        not_visited = sorted(not_visited, key=lambda x: self.graph_k.degree(x), reverse= True)
        nodes_tested = np.asarray(not_visited[:len(probs)-len(candidates)])
        nodes_tested = np.append(nodes_tested, candidates)
        self.update(nodes_tested)
        # self.params['epsilon'] = epsilon * (1/self.params['decay_factor'])
        # self.params['epsilon'] = epsilon ** self.params['decay_factor']
        self.params['epsilon'] = max(epsilon - epsilon/(self.params['decay_factor']+0.001), 0)
        test_result = self.current_results[1]
        self.params['rec_pos'] = np.asarray([nodes_tested[i] for i in range(len(nodes_tested)) if test_result[i]=='I'])
        self.params['visited'] = nodes_tested
 

    def get_result(self):
        return self.current_results

    def get_history(self):
        return self.history

    def get_graph_k(self):
        return self.graph_k

    def get_settings(self):
        settings = dict(method = self.method, graph_u = self.graph_u, spread = self.spread,
                    history_flag = self.history_flag, budget = self.budget)
        settings.update(self.params)
        return settings

    def get_states(self, mode='latest'):
        # from history, infer the latest state of the nodes
        if not self.history_flag and mode != 'latest':
            return ValueError(f'Cannot get states in mode {mode} when history is not activated for testing.')
        elif not self.history_flag:
            return {i[0]:i[1] for i in zip(self.current_results[0], self.current_results[1])}
        states = {}
        for item in self.history:
            for k in zip(item[0], item[1]):  # nodes: states
                states.setdefault(k[0], []).append(k[1])
        if mode == 'latest':
            return {k:v[-1] for k,v in states.items()}
        return states

    def get_latest_inf(self):
        return [i for i,j in zip(self.current_results[0], self.current_results[1]) if j =='I']
    
         


