import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import svds, eigs
import matplotlib.pyplot as plt
import snap
import utils
import tqdm

def analyze_degrees(graph):
    print("Analyzing degrees...")
    degreeList = [len(indices) for indices in graph.node2edge]
    degrees, freq = np.unique(degreeList, return_counts=True)
    with open("../results/{}_degrees.txt".format(graph.datatype), "w") as f:
        for _x, _y in zip(degrees, freq):
            f.write(f'{_x} {_y}\n')

    start_idx = int(degrees[0] == 0)
    utils.logAnalysis(degrees[start_idx:], freq[start_idx:],
                      "../plots/{}_degrees.png".format(graph.datatype),
                      "Degree", "Count", "Degrees")    
    print("The average degree of nodes:", np.mean(degreeList))
                  
def analyze_hyperedge_sizes(graph):
    print("Analyzing hyperedge sizes...")
    sizeList = [len(indices) for _, indices in graph.edges]
    sizes, freq = np.unique(sizeList, return_counts=True)
    with open("../results/{}_hyperedge_sizes.txt".format(graph.datatype), "w") as f:
        for _x, _y in zip(sizes, freq):
            f.write(f'{_x} {_y}\n')
    # draw log-log plot (y-axis: fraction of nodes, x-axis: degree)
    utils.logAnalysis(sizes, freq,
                      "../plots/{}_hyperedge_sizes.png".format(graph.datatype),
                      "Edge size", "Count", "Hyperedge Sizes")
    print("The average size of edges:", np.mean(sizeList))

def analyze_singular_values(graph, rank=100):
    print("Analyzing singular values...")
    incident_matrix, nnz = graph.get_incidence_matrix()
    _, s, _ = svds(incident_matrix.tocsc(), k=rank)
    s = sorted(s, reverse=True)
    with open("../results/{}_singular_values.txt".format(graph.datatype), "w") as f:
        for _x, _y in zip(np.arange(1,len(s)+1), s):
            f.write(f'{_x} {_y}\n')
    utils.logAnalysis(np.arange(1,len(s)+1), s,
                      "../plots/{}_singular_values.png".format(graph.datatype),
                      "Rank", "Singular value", "Singular Values")

def analyze_intersection(graph):
    print("Analyzing interesting pairs and intersection sizes...")
    inter_cnt, hyperedge_cnt = [0], [0]
    size_cnt = {}
    for i in tqdm.trange(1, graph.number_of_edges()):
        if graph.edges[i][0] != graph.edges[i-1][0]:
            inter_cnt.append(inter_cnt[-1])
            hyperedge_cnt.append(hyperedge_cnt[-1])
        hyperedge_cnt[-1] = i
        counter = {}
        for idx in graph.edges[i][1]:
            for edge_idx in graph.node2edge[graph.idx2node[idx]]:
                if edge_idx >= i: break
                counter[edge_idx] = counter.get(edge_idx, 0) + 1
        inter_cnt[-1] += len(counter)
        for v in counter.values():
            size_cnt[v] = size_cnt.get(v, 0) + 1
            
    inter_cnt = np.array(inter_cnt)
    hyperedge_cnt = np.array(hyperedge_cnt)

    with open("../results/{}_intersecting_pairs.txt".format(graph.datatype), "w") as f:
        for _x, _y in zip(hyperedge_cnt * (hyperedge_cnt + 1) / 2, inter_cnt):
            f.write(f'{_x} {_y}\n')
    
    start_idx = 0
    while inter_cnt[start_idx] == 0: start_idx += 1
    utils.logAnalysis((hyperedge_cnt * (hyperedge_cnt + 1) / 2)[start_idx:], inter_cnt[start_idx:],
                      "../plots/{}_intersecting_pairs.png".format(graph.datatype),
                      "# of all pairs", "# of intersecting pairs", "Intersecting Pairs")
    
    soi_keys, soi_values = zip(*sorted(size_cnt.items(), key=lambda x: x[0]))
    
    with open("../results/{}_intersection_sizes.txt".format(graph.datatype), "w") as f:
        for _x, _y in zip(soi_keys, soi_values):
            f.write(f'{_x} {_y}\n')
    
    start_idx = 0
    while soi_values[start_idx] == 0: start_idx += 1
    utils.logAnalysis(soi_keys[start_idx:], soi_values[start_idx:],
                      "../plots/{}_intersection_sizes.png".format(graph.datatype),
                      "Intersection size", "Count", "Intersection Sizes")
    
def analyze_edge_density(graph):
    print("Analyzing edge density...")
    keys = set([t for t, _ in graph.edges])
    if graph.datatype == 'model': keys.add(0)
    
    node_cnt = {k: 0 for k in keys}
    edge_cnt = {k: 0 for k in keys}
    for t, _ in graph.nodes: node_cnt[t] += 1
    for t, _ in graph.edges: edge_cnt[t] += 1
    node_cnt = [v for _, v in sorted(node_cnt.items(), key=lambda x: x[0])]
    edge_cnt = [v for _, v in sorted(edge_cnt.items(), key=lambda x: x[0])]
    for i in range(1, len(keys)):
        node_cnt[i] += node_cnt[i-1]
        edge_cnt[i] += edge_cnt[i-1]
    start_idx = 0
    while node_cnt[start_idx] == 0 or edge_cnt[start_idx] == 0: start_idx += 1
    with open("../results/{}_edge_density.txt".format(graph.datatype), "w") as f:
        for _d, _f in zip(node_cnt, edge_cnt):
            f.write(f'{_d} {_f}\n')
    utils.logAnalysis(node_cnt[start_idx:], edge_cnt[start_idx:],
                      "../plots/{}_edge_density.png".format(graph.datatype),
                      "# of nodes", "# of edges", "Edge Density")
    
def project2(graph, idx_ub):
    pg = snap.TUNGraph.New()
    nodeset = set([])
    for _, hyperedge in graph.edges[:idx_ub]:
        for n in hyperedge:
            if n not in nodeset:
                nodeset.add(n)
                pg.AddNode(n)

    for _, hyperedge in graph.edges[:idx_ub]:
        if len(hyperedge) == 1: continue
        for i in range(0, len(hyperedge)-1):
            for j in range(i+1, len(hyperedge)):
                i1, i2 = min(hyperedge[i], hyperedge[j]), max(hyperedge[i], hyperedge[j])
                ret = pg.AddEdge(i1, i2)
    return pg

def analyze_diameter_timestamp(graph):
    print("Analyzing effective diameter...")
    hyperedge_cnt = [0]
    for i in range(1, graph.number_of_edges()):
        if graph.edges[i][0] != graph.edges[i-1][0]:
            hyperedge_cnt.append(hyperedge_cnt[-1])
        hyperedge_cnt[-1] = i
    total_duration = len(hyperedge_cnt)
    
    timestamps = []
    diams = []
    with open("../results/{}_diameter.txt".format(graph.datatype), "w") as f:
        for i in range(total_duration):
            idx_ub = hyperedge_cnt[i]+1
            pg = project2(graph, idx_ub)
            _x = graph.edges[idx_ub-1][0]
            _y = snap.GetBfsEffDiam(pg, 4000 if i < 50 else 1000, False)
            f.write(f"{_x} {_y}\n")
            timestamps.append(_x)
            diams.append(_y)
            f.flush()
    plt.figure()
    plt.plot(timestamps, diams, 'ro')
    plt.xlabel("Time (year)")
    plt.ylabel("Effective Diameter")
    plt.title("Effective Diameter")
    plt.savefig("../plots/{}_diameter.png".format(graph.datatype), dpi=300)

def analyze_diameter_node(graph, total_duration):
    print("Analyzing effective diameter...")
    nodes = []
    diams = []
    with open("../results/{}_diameter.txt".format(graph.datatype), "w") as f:
        idx_ub = 0
        for i in range(total_duration):
            target_node_cnt = ((i+1) * graph.number_of_nodes()) // total_duration
            target_node_time = graph.nodes[target_node_cnt-1][0]
            while idx_ub < graph.number_of_edges() and graph.edges[idx_ub][0] <= target_node_time:
                idx_ub += 1
            pg = project2(graph, idx_ub)
            _x = pg.GetNodes()
            _y = snap.GetBfsEffDiam(pg, 4000 if i < 50 else 1000, False)
            f.write(f"{_x} {_y}\n")
            nodes.append(_x)
            diams.append(_y)
            f.flush()
    plt.figure()
    plt.plot(nodes, diams, 'ro')
    plt.xlabel("# of nodes")
    plt.ylabel("Effective Diameter")
    plt.title("Effective Diameter")
    plt.savefig("../plots/{}_diameter.png".format(graph.datatype), dpi=300)
    
def analyze_pattern(graph, sv_k=1000):
    analyze_degrees(graph)
    analyze_hyperedge_sizes(graph)
    analyze_intersection(graph)
    analyze_singular_values(graph, sv_k)
    analyze_edge_density(graph)
    if graph.datatype == 'coauth': analyze_diameter_timestamp(graph)
    else: analyze_diameter_node(graph, 200)
