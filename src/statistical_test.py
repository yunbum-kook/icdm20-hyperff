import powerlaw
class stat_test:    
    #1. Degree distribution
    def degreeTest(n2e):
        degrees = [len(indices) for indices in n2e]
        return powerlaw.Fit(degrees, discrete=True)
    
    #2. Edge size
    def sizeTest(e2n):
        sizes = [len(edge) for edge in e2n]
        return powerlaw.Fit(sizes, discrete=True)
    
    #3. Singular values
    def singTest(result_path):
        elems = []
        with open(result_path, "r") as f:
            for line in f:
                sv_key = int(line.strip().split()[0])
                sv_value = float(line.strip().split()[1])
                elems += [sv_key for _ in range(int(sv_value * 100))]
                elems.append(sv_value)
        return powerlaw.Fit(elems, discrete=True)
        
    #4. Size of Intersection
    def soiTest(result_path):
        elems = []
        with open(result_path, "r") as f:
            for line in f:
                soi_key = int(line.strip().split()[0])
                soi_value = int(line.strip().split()[1])
                elems += [soi_key for _ in range(soi_value)]
        return powerlaw.Fit(elems, discrete=True)
    
    # General Test
    def generalTest(fit):
        try:
            print("truncated power law vs exp", fit.distribution_compare('truncated_power_law', 'exponential', normalized_ratio = True))
            print("power law vs exp", fit.distribution_compare('power_law', 'exponential', normalized_ratio = True))
            print("lognormal vs exp", fit.distribution_compare('lognormal', 'exponential', normalized_ratio = True))
        except:
            print("statistical test failed...")
            
def statistical_test(graph):
    print("Running statistical test...")
    degreefit = stat_test.degreeTest([v for v in graph.node2edge])
    sizefit = stat_test.sizeTest([v for _, v in graph.edges])
    singfit = stat_test.singTest("../results/{}_singular_values.txt".format(graph.datatype))
    soifit = stat_test.soiTest("../results/{}_intersection_sizes.txt".format(graph.datatype))

    fits = [degreefit, sizefit, singfit, soifit]
    name = ["Degrees", "Hyperedge sizes", "Singular values", "Intersection Sizes"]
    
    for i in range(len(fits)):
        print(name[i])
        stat_test.generalTest(fits[i])
