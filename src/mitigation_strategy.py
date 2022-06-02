import networkx as nx
import numpy as np

class MitigationStrategy():
    def __init__(self, method, graph_known, graph_unknown, clusters, test_states, **params):
        # immutable
        self.method = method  # string
        self.clusters = dict(enumerate(clusters))  # {cluster_id: [members]}
        self.node_cluster = {i:k for k,v in self.clusters.items() for i in v}
        if params:
            self.params = params  # {restrict_time: T1, restrict_candidate_budget: M,
            #   restrict_candidate_neigh_budget: P, ban_time: T2 (or =T1)}
            if 'ban_time' not in self.params:
                self.params['ban_time'] = self.params['restrict_time']
        else:
            tmp = int(np.ceil(0.01*graph_known.number_of_nodes()))
            if tmp == 0:
                tmp = 5
            self.params = dict(restrict_time= 4, 
                            restrict_candidate_budget= tmp, 
                            restrict_candidate_neigh_budget= 4)
            self.params['ban_time'] = self.params['restrict_time']
        
        # mutable
        self.graph_k = graph_known.copy()  # updated by test startegy through set_graph_k
        self.graph_u = graph_unknown.copy()  # updated internally in mitigation strategy (frag)
        self.com_graphs = {k:nx.subgraph(self.graph_k,self.clusters[k]) for k in self.clusters}  # {cluster_id: cluster_graph}
        self.state = {n:'N' for n in self.graph_u.nodes()}  # 'N': not restricted, 'Y': restricted
        self.history = [self.state.copy()]  # keep history of the states
        self.ban = {}  # {node: time} dict of nodes banned to restrict, used in frag level, reduce time every 'run',
        #   remove from list once time=0
        self.restricted = {}  # {(nod1,node2): time} dict of edges restricted, reduce time every 'run', 
        #   put the edge back on once the time hits 0
        self.set_Cscore(test_states)  # sets self.Cscore {community: score}
        self.set_Nscore()  # sets self.Nscore {node: score} -- depends on self.Cscore
        
    def run(self, test_states, test_latest_inf):
        if self.method == 'commit':
            self.run_commit(test_states, test_latest_inf)
        else:
            raise NotImplementedError(f'Mitigation strategy {self.method} not defined.')
    
    def run_commit(self, test_states, test_latest_inf):
        '''
        input: 
            test_states: the state of nodes inferred by testing result over time {node:}
            test_latest_inf: the list of nodes that are identified as infected in this time [nodes_inf]
        '''
        # ----- update time for self.restricted and follow up graph_u, ban, and state update 
        # update time for already restricted edges
        self.restricted = {k: v-1 for k,v in self.restricted.items()}
        edge_release = [k for k,v in self.restricted.items() if v < 1]
        node_release = list(set([i for j in edge_release for i in j]))
        # update graph_u
        self.graph_u.add_edges_from(edge_release)
        # update ban and states and restricted
        for k in edge_release:
            self.restricted.pop(k)
        self.ban.update({n:self.params['ban_time'] for n in node_release})
        self.state.update({n:'N' for n in node_release})
        
        # ------ isolate new discovered infected
        edge_restrict = [(n,neigh) for n in test_latest_inf for neigh in self.graph_u.neighbors(n)]
        # update graph_u
        self.graph_u.remove_edges_from(edge_restrict)
        # update restricted, ban, states
        self.restricted.update({k:self.params['restrict_time'] for k in edge_restrict})
        for n in set(test_latest_inf).intersection(set(list(self.ban))):
            self.ban.pop(n)
        self.state.update({n:'Y' for n in test_latest_inf})
        
        # ------ calc cscore and nscore
        self.set_Cscore(test_states)
        self.set_Nscore()
        
        # ----- frag
        sorted_scores = {k: v for k, v in sorted(self.Nscore.items(), key=lambda item: item[1])}
        # pick top M with P neight, not in ban --> edges_rem
        candidates = list(sorted_scores.keys())[:self.params['restrict_candidate_budget']]
        for node in candidates:
            nn = list(self.graph_k.neighbors(node))
            if len(nn) == 0:
                continue
            if len(nn) > self.params['restrict_candidate_neigh_budget']:
                neighbors = list(np.random.choice(nn,
                            size=self.params['restrict_candidate_neigh_budget'],replace=False)) 
            else:
                neighbors = nn
            rem_cand = [node]+neighbors
            self.state.update({n:'Y' for n in rem_cand})
            subg = nx.subgraph(self.graph_k,rem_cand)
            edges_all = [list(self.graph_k.edges(n)) for n in rem_cand]
            edges_all = [j for i in edges_all for j in i if j[0] not in self.ban and self.state[j[0]] == 'N' \
                         and j[1] not in self.ban and self.state[j[1]] == 'N']
            edges_rem = [e for e in edges_all if e not in subg.edges()]  # keep edges inside and isolate from outside
            # update graph_u
            self.graph_u.remove_edges_from(edges_rem)
            # update restricted, state
            self.restricted.update({k:self.params['restrict_time'] for k in edges_rem})
        
        # --- update history
        self.history.append(self.state.copy())
        
    def get_community_score(self, com_graph, test_states):
        '''
        Cscore = (normalized size + infected ratio + separability)/(1+1+1)
        separability = (intra_neighbor_edges - inter_neighbor_edges)/all edges in the graph
        NOTE: changes in com_graph and graph_k are linked! (com_graph is subgraph of graph_k)
        '''
        # normalized size
        cnodes = com_graph.number_of_nodes()
        normalized_size = cnodes/self.graph_k.number_of_nodes()
        # infected ratio
        infected = [n for n in com_graph.nodes() if test_states.get(n, None)=='I']
        infected_ratio = len(infected)/cnodes
        # separability
        if self.graph_k.number_of_edges() == 0:
            return (normalized_size+infected_ratio)/2
        edges_inside = com_graph.number_of_edges()
        edges_outside = sum([self.graph_k.degree(n) - com_graph.degree(n) for n in com_graph])
        separability = (edges_inside - edges_outside)/self.graph_k.number_of_edges()
        return (normalized_size+infected_ratio+separability)/3
    
    def set_Cscore(self, test_states):
        self.Cscore = {k:self.get_community_score(self.com_graphs[k], test_states) for k in self.com_graphs}
        
    def get_neighborhood_connectivity(self,node):
        # edges between neighbors/(edges between neighbors + edges from neighbors to outside)
        nodes = [node] + [i for i in self.graph_k.neighbors(node)]
        if len(nodes) == 1:  #-----------------------> we can make isolated nodes among unvisited nodes
            return 0
        subgraph = nx.subgraph(self.graph_k,nodes)
        # only count edges between neighbors and not to the node
        edges_inside = subgraph.number_of_edges() - subgraph.degree(node) # 0 if neighbors not connected at all
        edges_outside = sum([self.graph_k.degree(n) - subgraph.degree(n) for n in subgraph if n != node])
        if edges_inside ==0 and edges_outside ==0: # e.g, node has one neightbor and neighbor has only the one edge to the node
            return 0
        return edges_inside/(edges_inside + edges_outside)
    
    def set_Nscore(self):
        '''
        this is the restriction score
        has to be redone every iteration in commit original algorithm and with commit knowledge
        this, in commit original alg, is from the simulation perspective just to compare how 
            off we are in commit network
        nscore = (cscore (of the community of the node) + neigh_connectivity)/2 
            if node not in ban, or already restricted
        '''
        self.Nscore = {node: (self.Cscore[self.node_cluster[node]]+ \
                        self.get_neighborhood_connectivity(node))/2 for \
                        node in self.graph_k if node not in self.ban and self.state[node] == 'N'}
        
    def get_graph_u(self):
        return self.graph_u
        
    def set_graph_k(self, g):
        self.graph_k = g
        # update com_graphs
        self.com_graphs = {k:nx.subgraph(self.graph_k,self.clusters[k]) for k in self.clusters}

    def get_history(self):
        return self.history
        