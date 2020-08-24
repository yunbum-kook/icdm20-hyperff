import os
import sys
import argparse

from hypergraph import Hypergraph, NullHypergraph
from model import HyperFF
from analysis_pattern import analyze_pattern
from analysis_decomposition import analyze_decomposition
from statistical_test import statistical_test

def main(graph):
    sv_k = {'contact': 326, 'email': 1000, 'substances': 5000, 'tags': 3000, 'threads': 1000, 'coauth': 500, 'model': 500}
    print('# Structural patterns and dynamical patterns')
    print('(See ../results/{}_*.txt and ../plots/{}_*.txt)'.format(graph.datatype, graph.datatype))
    analyze_pattern(graph, sv_k[graph.datatype])
    if graph.datatype == 'substances':
        print('\n## Comparision with null model')
        _graph = NullHypergraph(graph)
        analyze_pattern(_graph, 500)
    if graph.datatype in ('substances', 'model'):
        print('\n# Patterns in decomposed graph')
        analyze_decomposition(graph)
    print('\n# Statistical test (three heavy-tailed distributions versus exponential distribution)')
    statistical_test(graph)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Official Implementation for 'Evolution of Real-world Hypergraphs: Patterns and Models without Oracles'")
    parser.add_argument('dataset', type=str, help='Select dataset for analysis')
    
    parser.add_argument(
        "-p",
        "--burning",
        action="store",
        default=0.51,
        type=float,
        help="Select the burning probability p (if the target dataset is 'model')",
    )
    parser.add_argument(
        "-q",
        "--expanding",
        action="store",
        default=0.2,
        type=float,
        help="Select the expanding probability q  (if the target dataset is 'model')",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        action="store",
        default=10000,
        type=int,
        help="Select the number of nodes n (if the target dataset is 'model')",
    )
    args = parser.parse_args()
    datasets_info = {'contact': 'contact-high-school',
                     'email': 'email-Eu-full',
                     'substances': 'NDC-substances-full',
                     'tags': 'tags-ask-ubuntu',
                     'threads': 'threads-math-sx',
                     'coauth': 'coauth-DBLP-full'}

    if not os.path.exists('../results'):
        os.mkdir('../results')
    if not os.path.exists('../plots'):
        os.mkdir('../plots')
    if args.dataset in datasets_info:
        graph = Hypergraph(datasets_info[args.dataset], args.dataset)
    elif args.dataset == 'model':
        print("Generating hypergraph using HyperFF model...")
        graph = HyperFF(args.burning, args.expanding, args.nodes - 1)
    else:
        print("Invalid arguments.")
        parser.print_help()
        sys.exit(0)
    main(graph)
