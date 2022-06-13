import numpy as np
import pickle
from collections import Counter
import networkx as nx

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    G_u = data['graph'] # graph unknown = original graph
    clusters = data['coms']
    G_k = nx.Graph()  # graph known = the one we fill it up
    G_k.add_nodes_from(G_u.nodes())
    sources = data['sources']
    return G_u, G_k, sources, clusters


# mit metric 
def calc_mit_metric(mhist, graph):
    '''
    input: 
        - mitigation history: [{node: Y or N}], len = #runs of mit + 1
        - graph: orginal graph 
    output: 
        - number of nodes in the restricted zone ('Y' state) at each time (new and old)/total number of graph nodes
            len = #runs of mit + 1
    '''
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0: 
        raise ValueError('The input graph is empty!')
    return np.asarray([len([k for k,v in item.items() if v == 'Y'])/num_nodes for item in mhist])

# test metric
def calc_test_metric(thist, shist):
    '''
    input: 
        - test history: [([nodes],[states],{graph info})], len = #runs of test
        - spread history: # [{node: SIR}], len = #runs of spread + 1
        Note: since test is called before spread in pipeline, the last thist item is for shist[-2]
    output: 
        - efficiency: [test_inf/test_budget], len = #runs of test
        - efficacy: [test_inf/all_inf], len = #runs of test (or len shist - 1)
    '''
    efficiency = [dict(Counter(obs[1])).get('I', 0)/len(obs[0]) for obs in thist]
    efficacy = []
    for tst,spr in zip(thist,shist[:-1]):
        tot_pos = dict(Counter(list(spr.values()))).get('I')
        if tot_pos > 0:
            efficacy.append(dict(Counter(tst[1])).get('I', 0)/tot_pos)
        else:
            efficacy.append(0)
    return {'efficiency': np.asarray(efficiency), 'efficacy': np.asarray(efficacy)}

# spread metric
def calc_spread_metric(shist, graph):
    '''
    input:
        - spread history: # [{node: SIR}], len = #runs of spread + 1
        - graph: orginal graph 
    output:
        - new_inf: [#new_inf/#graph_nodes], len = #runs of spread + 1
        - duration: duration of spread, len = 1
        - inf per stamp on avg: sigma(inf)/duration, len =1
    '''
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0: 
        raise ValueError('The input graph is empty!')
    new_inf = [len([node for node,state in shist[0].items() if state == 'I'])]
    new_inf = np.asarray(new_inf + [len([node for node,state in shist[idx].items() \
               if state == 'I' and shist[idx-1][node] != 'I']) for idx in range(1, len(shist))])/num_nodes
    duration = len(shist)
    if duration == 0: 
        raise ValueError('The input history for spread is empty!')
    avg_per_stamp = np.sum(new_inf)/duration
    return {'new_inf': new_inf, 'duration': duration, 'avg_per_stamp': avg_per_stamp}
    