from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds, eigs
import numpy as np
from itertools import chain
import random

class Hypergraph:
    def __init__(self, name, datatype=None):
        self.name = name
        self.datatype = datatype
        if datatype is not None:
            self.f_nverts = open("../data/" + self.name + "/" + self.name + "-nverts.txt", 'r')
            self.f_simplices = open("../data/" + self.name + "/" + self.name + "-simplices.txt", 'r')
            self.f_times = open("../data/" + self.name + "/" + self.name + "-times.txt", 'r')
            self.load_graph()
            self.f_nverts.close()
            self.f_simplices.close()
            self.f_times.close()
        
    def number_of_nodes(self): return len(self.nodes)
    def number_of_edges(self): return len(self.edges)
    
    def load_graph(self):
        def load_hyperedge(_size, _time):
            return [int(self.f_simplices.readline()) for _ in range(_size)]
            
        self.incidence_list = []
        if self.datatype is None: return
        
        raw_hyperedges = [(int(_time), load_hyperedge(int(_size), _time)) for i, (_size, _time) in enumerate(zip(self.f_nverts, self.f_times))]
        # Sorts are guaranteed to be stable: https://docs.python.org/ko/3.6/howto/sorting.html
        self.edges = sorted(raw_hyperedges, key=lambda x: x[0])
        
        self.idx2node = {}
        self.nodes = []
        self.node2edge = []
        for i, (_time, hyperedge) in enumerate(self.edges):
            for idx in hyperedge:
                if idx not in self.idx2node:
                    self.idx2node[idx] = len(self.nodes)
                    self.nodes.append((_time, idx))
                    self.node2edge.append([])
                self.node2edge[self.idx2node[idx]].append(i)
    
    def get_incidence_matrix(self):
        rows, cols = zip(*chain.from_iterable([[(i, edge_idx) for edge_idx in edges]
                                               for i, edges in enumerate(self.node2edge)]))
        nnz = len(rows)
        return coo_matrix((np.ones(nnz), (rows, cols)), shape=(self.number_of_nodes(), self.number_of_edges())), nnz
    
    
class NullHypergraph(Hypergraph):
    def __init__(self, graph, random_seed=1000):
        super(NullHypergraph, self).__init__(graph.name, None)
        self.edges = []
        self.idx2node = {}
        self.nodes = []
        self.node2edge = []
        
        self.original_graph = graph
        self.gen_null_model(random_seed)
        self.datatype = 'null'
        
    def add_edges(self, hyperedge, timestamp):
        assert len(self.edges) == 0 or timestamp >= self.edges[-1][0]
        new_idx = len(self.edges)
        self.edges.append((timestamp, hyperedge))
        for idx in hyperedge:
            if idx not in self.idx2node:
                self.idx2node[idx] = len(self.nodes)
                self.nodes.append((timestamp, idx))
                self.node2edge.append([])
            self.node2edge[self.idx2node[idx]].append(new_idx)
     
    def gen_null_model(self, random_seed):
        random.seed(random_seed)
        np.random.seed(random_seed)
        graph = self.original_graph
        n = graph.number_of_nodes()
        node_indices = [_ for _ in range(n)]
        nodeset = set([])
        for i in range(graph.number_of_edges()):
            timestamp = graph.edges[i][0]
            for _n in graph.edges[i][1]:
                nodeset.add(_n)
            for j in range(len(graph.edges[i][1])):
                target_idx = np.random.randint(len(nodeset) - j) + j
                node_indices[j], node_indices[target_idx] = node_indices[target_idx], node_indices[j]
            self.add_edges(node_indices[:len(graph.edges[i][1])], timestamp)
