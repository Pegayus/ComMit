import networkx as nx
import numpy as np
import pdb

class MitigationStrategy():
    def __init__(self, method, graph_known, graph_unknown, clusters, test_states, **params):
        # immutable
        self.method = method  # string
        self.clusters = dict(enumerate(clusters))  # {cluster_id: [members]}
        self.node_cluster = {i:k for k,v in self.clusters.items() for i in v}
        if params:
            self.params = params  # {restrict_time: T1, restrict_candidate_budget: M,
            #   restrict_candidate_neigh_budget: P, ban_time: T2 (or =T1), cthr:}
            if 'ban_time' not in self.params:
                self.params['ban_time'] = self.params['restrict_time']
            if self.params['restrict_candidate_budget'] < 1:
                self.params['restrict_candidate_budget'] = 5
            if self.params['restrict_candidate_neigh_budget'] < 1:
                self.params['restrict_candidate_neigh_budget'] = 2
        else:
            tmp = int(np.ceil(0.001*graph_known.number_of_nodes()))
            if tmp == 0:
                tmp = 5
            self.params = dict(restrict_time= 4, 
                            restrict_candidate_budget= tmp, 
                            restrict_candidate_neigh_budget= 2, 
                            community_thr = 0.1)
            self.params['ban_time'] = self.params['restrict_time']
        
        # mutable
        self.graph_k = graph_known.copy()  # updated by test startegy through set_graph_k
        self.graph_u = graph_unknown.copy()  # updated internally in mitigation strategy (frag)
        self.com_graphs = {k:nx.subgraph(self.graph_k,self.clusters[k]) for k in self.clusters}  # {cluster_id: cluster_graph}
        self.state = {n:'N' for n in self.graph_u.nodes()}  # 'N': not restricted, 'Y': restricted
        self.history = [self.state.copy()]  # keep history of the states
        self.ban = {}  # {node: time} dict of nodes banned to restrict, used in frag level, reduce time every 'run',
        #   remove from list once time=0
        self.restricted_node = {}  # {node: time} dict of nodes restricted, reduce time every 'run', 
        #   release node once the time hits 0
        self.restricted_edge = {}  # {(nod1,node2): time} dict of edges restricted, reduce time every 'run', 
        #   put the edge back on once the time hits 0
        self.set_Cscore(test_states)  # sets self.Cscore {community: score}
        self.set_Nscore()  # sets self.Nscore {node: score} -- depends on self.Cscore
        
    def run(self, test_states, test_latest_inf):
        if self.method == 'commit':
            self.run_commit(test_states, test_latest_inf)
        elif self.method == 'comiso':
            self.run_comiso(test_states, test_latest_inf)
        elif self.method == 'degiso':
            self.run_degiso(test_states, test_latest_inf)
        elif self.method == '1hopiso':
            self.run_1hopiso(test_states, test_latest_inf)
        elif self.method == 'random':
            self.run_random(test_states, test_latest_inf)
        else:
            raise NotImplementedError(f'Mitigation strategy {self.method} not defined.')
    
    def run_random(self, test_states, test_latest_inf):
        '''
        randomly choose the seed node and its neighbors (acquaintance immunization method)
        input: 
            test_states: the state of nodes inferred by testing result over time {node:}
            test_latest_inf: the list of nodes that are identified as infected in this time [nodes_inf]
        '''
        # ----- update time for self.restricted and self.ban and follow up graph_u, ban, and state update 
        # update time for already banned nodes
        self.ban = {k: v-1 for k,v in self.ban.items()}
        rem = [k for k,v in self.ban.items() if v<1]
        for k in rem:
            self.ban.pop(k)
        # update time for already restricted edges
        self.restricted_edge = {k: v-1 for k,v in self.restricted_edge.items()}
        self.restricted_node = {k: v-1 for k,v in self.restricted_node.items()}
        edges = [k for k,v in self.restricted_edge.items() if v < 1]  
        node_release = [k for k,v in self.restricted_node.items() if v < 1]
        edge_release = []
        # we don't want the edges to be released if the other end is still in restriction
        for e in edges: 
            if e[0] in node_release:
                if self.state[e[1]] == 'N':
                    edge_release.append(e)
            else:
                if self.state[e[0]] == 'N':
                    edge_release.append(e)
        # update graph_u
        self.graph_u.add_edges_from(edge_release)
        # update ban and states and restricted
        for k in edges:
            self.restricted_edge.pop(k)
        for k in node_release:
            self.restricted_node.pop(k)
        self.ban.update({n:self.params['ban_time'] for n in node_release})  # -- when does it count down and remove?
        self.state.update({n:'N' for n in node_release})

        
        # ------ isolate new discovered infected
        # rest_node_update = {n: self.params['restrict_time'] for n in test_latest_inf}
        # rest_edge_update = {(n,neigh): self.params['restrict_time'] for n in test_latest_inf for neigh in self.graph_u.neighbors(n)}
        # # update graph_u
        # self.graph_u.remove_edges_from(list(rest_edge_update))
        # # update restricted, ban, states
        # self.restricted_edge.update(rest_edge_update)
        # self.restricted_node.update(rest_node_update)
        # self.state.update({n:'Y' for n in rest_node_update})
        # for n in set(list(rest_node_update)).intersection(set(list(self.ban))):
        #     self.ban.pop(n)

    
        
        # ----- frag
        candidates = [node for node in self.graph_k if node not in self.ban]
        if self.params['restrict_candidate_budget'] < len(candidates):
            candidates = np.random.choice(candidates, self.params['restrict_candidate_budget'], replace=False)
        else:
            candidate = np.asarray(candidate)
        for node in candidates:
            # decision based on the graph we know
            nn = list(self.graph_k.neighbors(node))
            if len(nn) == 0:
                continue
            if len(nn) > self.params['restrict_candidate_neigh_budget']:
                neighbors = list(np.random.choice(nn,
                            size=self.params['restrict_candidate_neigh_budget'],replace=False)) 
            else:
                neighbors = nn
            # severing ties in the graph that we don't know based on the node restrictions
            rem_cand = [node]+neighbors
            if len(set(rem_cand)) < 2:
                continue
            subg = nx.subgraph(self.graph_u, rem_cand)

            rest_node_update = {n: self.params['restrict_time'] for n in rem_cand}
            rest_edge_update = {i: self.params['restrict_time'] for n in rem_cand for i in list(self.graph_u.edges(n)) if \
                                    n not in self.ban}
            edges_all = list(rest_edge_update)
            edges_rem = [e for e in edges_all if e not in subg.edges()]  # keep edges inside and isolate from outside
            rest_edge_update = {k:v for k,v in rest_edge_update.items() if k in edges_rem}
            # update graph_u
            self.graph_u.remove_edges_from(edges_rem)
            # update restricted, state
            self.restricted_node.update(rest_node_update)
            self.restricted_edge.update(rest_edge_update)
            self.state.update({n:'Y' for n in rest_node_update})


        # --- update history
        self.history.append(self.state.copy())
     


    def run_comiso(self, test_states, test_latest_inf):
        '''
        input: 
            test_states: the state of nodes inferred by testing result over time {node:}
            test_latest_inf: the list of nodes that are identified as infected in this time [nodes_inf]
        '''
        # ----- update time for self.restricted and follow up graph_u, ban, and state update 
        # update time for already restricted edges
        self.restricted_edge = {k: v-1 for k,v in self.restricted_edge.items()}
        self.restricted_node = {k: v-1 for k,v in self.restricted_node.items()}
        edges = [k for k,v in self.restricted_edge.items() if v < 1]  
        node_release = [k for k,v in self.restricted_node.items() if v < 1]
        edge_release = []
        # we don't want the edges to be released if the other end is still in restriction
        for e in edges: 
            if e[0] in node_release:
                if self.state[e[1]] == 'N':
                    edge_release.append(e)
            else:
                if self.state[e[0]] == 'N':
                    edge_release.append(e)
        # update graph_u
        self.graph_u.add_edges_from(edge_release)
        # update ban and states and restricted
        for k in edges:
            self.restricted_edge.pop(k)
        for k in node_release:
            self.restricted_node.pop(k)
        self.state.update({n:'N' for n in node_release})


        # ------ isolate new discovered infected
        # rest_node_update = {n: self.params['restrict_time'] for n in test_latest_inf}
        # rest_edge_update = {(n,neigh): self.params['restrict_time'] for n in test_latest_inf for neigh in self.graph_u.neighbors(n)}
        # # update graph_u
        # self.graph_u.remove_edges_from(list(rest_edge_update))
        # # update restricted, ban, states
        # self.restricted_edge.update(rest_edge_update)
        # self.restricted_node.update(rest_node_update)
        # self.state.update({n:'Y' for n in rest_node_update})

        # ------ calc comscore and conditional isolation
        for members in self.clusters.values():
            num_inf = len([k for k in members if test_states.get(k,None)=='I'])
            if num_inf/len(members) >= self.params['community_thr']:
                rem_cand = members
                if len(set(rem_cand)) < 2:
                    continue
                subg = nx.subgraph(self.graph_u, rem_cand)

                rest_node_update = {n: self.params['restrict_time'] for n in rem_cand}
                rest_edge_update = {i: self.params['restrict_time'] for n in rem_cand for i in list(self.graph_u.edges(n)) if \
                                        self.state[n] == 'N'}
                edges_all = list(rest_edge_update)
                edges_rem = [e for e in edges_all if e not in subg.edges()]  # keep edges inside and isolate from outside
                rest_edge_update = {k:v for k,v in rest_edge_update.items() if k in edges_rem}
                # update graph_u
                self.graph_u.remove_edges_from(edges_rem)
                # update restricted, state
                self.restricted_node.update(rest_node_update)
                self.restricted_edge.update(rest_edge_update)
                self.state.update({n:'Y' for n in rest_node_update})
                
        # --- update history
        self.history.append(self.state.copy())

    def run_degiso(self, test_states, test_latest_inf):
        '''
        input: 
            test_states: the state of nodes inferred by testing result over time {node:}
            test_latest_inf: the list of nodes that are identified as infected in this time [nodes_inf]
        '''
        # ----- update time for self.restricted and self.ban and follow up graph_u, ban, and state update 
        # update time for already banned nodes
        self.ban = {k: v-1 for k,v in self.ban.items()}
        rem = [k for k,v in self.ban.items() if v<1]
        for k in rem:
            self.ban.pop(k)
        # update time for already restricted edges
        self.restricted_edge = {k: v-1 for k,v in self.restricted_edge.items()}
        self.restricted_node = {k: v-1 for k,v in self.restricted_node.items()}
        edges = [k for k,v in self.restricted_edge.items() if v < 1]  
        node_release = [k for k,v in self.restricted_node.items() if v < 1]
        edge_release = []
        # we don't want the edges to be released if the other end is still in restriction
        for e in edges: 
            if e[0] in node_release:
                if self.state[e[1]] == 'N':
                    edge_release.append(e)
            else:
                if self.state[e[0]] == 'N':
                    edge_release.append(e)
        # update graph_u
        self.graph_u.add_edges_from(edge_release)
        # update ban and states and restricted
        for k in edges:
            self.restricted_edge.pop(k)
        for k in node_release:
            self.restricted_node.pop(k)
        self.ban.update({n:self.params['ban_time'] for n in node_release})  # -- when does it count down and remove?
        self.state.update({n:'N' for n in node_release})

        
        # ------ isolate new discovered infected
        rest_node_update = {n: self.params['restrict_time'] for n in test_latest_inf}
        rest_edge_update = {(n,neigh): self.params['restrict_time'] for n in test_latest_inf for neigh in self.graph_u.neighbors(n)}
        # update graph_u
        self.graph_u.remove_edges_from(list(rest_edge_update))
        # update restricted, ban, states
        self.restricted_edge.update(rest_edge_update)
        self.restricted_node.update(rest_node_update)
        self.state.update({n:'Y' for n in rest_node_update})
        for n in set(list(rest_node_update)).intersection(set(list(self.ban))):
            self.ban.pop(n)

        
        # ------ calc degscore
        degscore = {node:self.graph_k.degree(node) for node in self.graph_k}
        
        # ----- frag
        thr = np.percentile(list(degscore.values()), 80)
        sorted_scores = sorted([(k,v) for k,v in degscore.items() if v >= thr], key=lambda item: item[1], reverse=True)
        # pick top M with P neight, not in ban --> edges_rem
        candidates = [i[0] for i in sorted_scores[:self.params['restrict_candidate_budget']]]
        for node in candidates:
            # decision based on the graph we know
            nn = list(self.graph_k.neighbors(node))
            if len(nn) == 0:
                continue
            if len(nn) > self.params['restrict_candidate_neigh_budget']:
                neighbors = list(np.random.choice(nn,
                            size=self.params['restrict_candidate_neigh_budget'],replace=False)) 
            else:
                neighbors = nn
            # severing ties in the graph that we don't know based on the node restrictions
            rem_cand = [node]+neighbors
            if len(set(rem_cand)) < 2:
                continue
            subg = nx.subgraph(self.graph_u, rem_cand)

            rest_node_update = {n: self.params['restrict_time'] for n in rem_cand}
            rest_edge_update = {i: self.params['restrict_time'] for n in rem_cand for i in list(self.graph_u.edges(n)) if \
                                    n not in self.ban and self.state[n] == 'N'}
            edges_all = list(rest_edge_update)
            edges_rem = [e for e in edges_all if e not in subg.edges()]  # keep edges inside and isolate from outside
            rest_edge_update = {k:v for k,v in rest_edge_update.items() if k in edges_rem}
            # update graph_u
            self.graph_u.remove_edges_from(edges_rem)
            # update restricted, state
            self.restricted_node.update(rest_node_update)
            self.restricted_edge.update(rest_edge_update)
            self.state.update({n:'Y' for n in rest_node_update})


        # --- update history
        self.history.append(self.state.copy())
     
    def run_1hopiso(self, test_states, test_latest_inf):
        '''
        input: 
            test_states: the state of nodes inferred by testing result over time {node:}
            test_latest_inf: the list of nodes that are identified as infected in this time [nodes_inf]
        '''
        # ----- update time for self.restricted and self.ban and follow up graph_u, ban, and state update 
        # update time for already restricted edges
        self.restricted_edge = {k: v-1 for k,v in self.restricted_edge.items()}
        self.restricted_node = {k: v-1 for k,v in self.restricted_node.items()}
        edges = [k for k,v in self.restricted_edge.items() if v < 1]  
        node_release = [k for k,v in self.restricted_node.items() if v < 1]
        edge_release = []
        # we don't want the edges to be released if the other end is still in restriction
        for e in edges: 
            if e[0] in node_release:
                if self.state[e[1]] == 'N':
                    edge_release.append(e)
            else:
                if self.state[e[0]] == 'N':
                    edge_release.append(e)
        # update graph_u
        self.graph_u.add_edges_from(edge_release)
        # update ban and states and restricted
        for k in edges:
            self.restricted_edge.pop(k)
        for k in node_release:
            self.restricted_node.pop(k)
        self.state.update({n:'N' for n in node_release})

        
        # ------ isolate new discovered infected
        # rest_node_update = {n: self.params['restrict_time'] for n in test_latest_inf}
        # rest_edge_update = {(n,neigh): self.params['restrict_time'] for n in test_latest_inf for neigh in self.graph_u.neighbors(n)}
        # # update graph_u
        # self.graph_u.remove_edges_from(list(rest_edge_update))
        # # update restricted, ban, states
        # self.restricted_edge.update(rest_edge_update)
        # self.restricted_node.update(rest_node_update)
        # self.state.update({n:'Y' for n in rest_node_update})

        
        # ------ calc node score based on inf status (new_inf)
        degscore = {node:self.graph_u.degree(node) for node in test_latest_inf}
        
        # ----- frag
        sorted_scores = sorted([(k,v) for k,v in degscore.items()], key=lambda item: item[1], reverse=True)
        # pick top M with P neight, not in ban --> edges_rem
        candidates = [i[0] for i in sorted_scores[:self.params['restrict_candidate_budget']]]
        for node in candidates:
            # decision based on the graph we know
            nn = list(self.graph_k.neighbors(node))
            if len(nn) == 0:
                continue
            if len(nn) > self.params['restrict_candidate_neigh_budget']:
                neighbors = list(np.random.choice(nn,
                            size=self.params['restrict_candidate_neigh_budget'],replace=False)) 
            else:
                neighbors = nn
            # severing ties in the graph that we don't know based on the node restrictions
            rem_cand = [node]+neighbors
            if len(set(rem_cand)) < 2:
                continue
            subg = nx.subgraph(self.graph_u, rem_cand)

            rest_node_update = {n: self.params['restrict_time'] for n in rem_cand}
            rest_edge_update = {i: self.params['restrict_time'] for n in rem_cand for i in list(self.graph_u.edges(n)) if \
                                    self.state[n] == 'N'}
            edges_all = list(rest_edge_update)
            edges_rem = [e for e in edges_all if e not in subg.edges()]  # keep edges inside and isolate from outside
            rest_edge_update = {k:v for k,v in rest_edge_update.items() if k in edges_rem}
            # update graph_u
            self.graph_u.remove_edges_from(edges_rem)
            # update restricted, state
            self.restricted_node.update(rest_node_update)
            self.restricted_edge.update(rest_edge_update)
            self.state.update({n:'Y' for n in rest_node_update})


        # --- update history
        self.history.append(self.state.copy())
      
    def run_commit(self, test_states, test_latest_inf):
        '''
        input: 
            test_states: the state of nodes inferred by testing result over time {node:}
            test_latest_inf: the list of nodes that are identified as infected in this time [nodes_inf]
        '''
        # ----- update time for self.restricted and self.ban and follow up graph_u, ban, and state update 
        # update time for already banned nodes
        self.ban = {k: v-1 for k,v in self.ban.items()}
        rem = [k for k,v in self.ban.items() if v<1]
        for k in rem:
            self.ban.pop(k)
        # update time for already restricted edges
        self.restricted_edge = {k: v-1 for k,v in self.restricted_edge.items()}
        self.restricted_node = {k: v-1 for k,v in self.restricted_node.items()}
        edges = [k for k,v in self.restricted_edge.items() if v < 1]  
        node_release = [k for k,v in self.restricted_node.items() if v < 1]
        edge_release = []
        # we don't want the edges to be released if the other end is still in restriction
        for e in edges: 
            if e[0] in node_release:
                if self.state[e[1]] == 'N':
                    edge_release.append(e)
            else:
                if self.state[e[0]] == 'N':
                    edge_release.append(e)
        # update graph_u
        self.graph_u.add_edges_from(edge_release)
        # update ban and states and restricted
        for k in edges:
            self.restricted_edge.pop(k)
        for k in node_release:
            self.restricted_node.pop(k)
        self.ban.update({n:self.params['ban_time'] for n in node_release})  # -- when does it count down and remove?
        self.state.update({n:'N' for n in node_release})

        
        # ------ isolate new discovered infected
        rest_node_update = {n: self.params['restrict_time'] for n in test_latest_inf}
        rest_edge_update = {(n,neigh): self.params['restrict_time'] for n in test_latest_inf for neigh in self.graph_u.neighbors(n)}
        # update graph_u
        self.graph_u.remove_edges_from(list(rest_edge_update))
        # update restricted, ban, states
        self.restricted_edge.update(rest_edge_update)
        self.restricted_node.update(rest_node_update)
        self.state.update({n:'Y' for n in rest_node_update})
        for n in set(list(rest_node_update)).intersection(set(list(self.ban))):
            self.ban.pop(n)

        
        # ------ calc cscore and nscore
        self.set_Cscore(test_states)
        self.set_Nscore()
        
        # ----- frag
        thr = np.percentile(list(self.Nscore.values()), 80)
        sorted_scores = sorted([(k,v) for k,v in self.Nscore.items() if v >= thr], key=lambda item: item[1], reverse=True)
        # pick top M with P neight, not in ban --> edges_rem
        candidates = [i[0] for i in sorted_scores[:self.params['restrict_candidate_budget']]]
        for node in candidates:
            # decision based on the graph we know
            nn = list(self.graph_k.neighbors(node))
            if len(nn) == 0:
                continue
            if len(nn) > self.params['restrict_candidate_neigh_budget']:
                neighbors = list(np.random.choice(nn,
                            size=self.params['restrict_candidate_neigh_budget'],replace=False)) 
            else:
                neighbors = nn
            # severing ties in the graph that we don't know based on the node restrictions
            rem_cand = [node]+neighbors
            if len(set(rem_cand)) < 2:
                continue
            subg = nx.subgraph(self.graph_u, rem_cand)

            rest_node_update = {n: self.params['restrict_time'] for n in rem_cand}
            rest_edge_update = {i: self.params['restrict_time'] for n in rem_cand for i in list(self.graph_u.edges(n)) if \
                                    n not in self.ban and self.state[n] == 'N'}
            edges_all = list(rest_edge_update)
            edges_rem = [e for e in edges_all if e not in subg.edges()]  # keep edges inside and isolate from outside
            rest_edge_update = {k:v for k,v in rest_edge_update.items() if k in edges_rem}
            # update graph_u
            self.graph_u.remove_edges_from(edges_rem)
            # update restricted, state
            self.restricted_node.update(rest_node_update)
            self.restricted_edge.update(rest_edge_update)
            self.state.update({n:'Y' for n in rest_node_update})


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
        # A: self.Nscore = {node: self.graph_k.degree(node)/self.graph_k.number_of_nodes() for \
        #                 node in self.graph_k if node not in self.ban and self.state[node] == 'N'}
        # B: 
        self.Nscore = {node: self.Cscore[self.node_cluster[node]] for \
                        node in self.graph_k if node not in self.ban and self.state[node] == 'N'}
        # C: self.Nscore = {node: (self.Cscore[self.node_cluster[node]]+ \
        #                 self.get_neighborhood_connectivity(node))/2 for \
        #                 node in self.graph_k if node not in self.ban and self.state[node] == 'N'}
        # NOTE: B > A > C in terms of duration of infection, in terms of budget, they are almost similar.
        
    def get_graph_u(self):
        return self.graph_u
        
    def set_graph_k(self, g):
        self.graph_k = g
        # update com_graphs
        self.com_graphs = {k:nx.subgraph(self.graph_k,self.clusters[k]) for k in self.clusters}

    def get_history(self):
        return self.history
        