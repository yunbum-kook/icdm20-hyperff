import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

def powerfit(x, y):
    """line fitting on log-log scale"""
    k, m = np.polyfit(np.log(x[:]), np.log(y), 1)
    return np.exp(m) * x**(k)

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, r_value**2

def logAnalysis(xaxis, yaxis, filename, xname="x axis", yname="y axis", title="title"):
    regression = powerfit(xaxis, yaxis)

    plt.figure()
    plt.loglog(xaxis, yaxis, 'ro')
    plt.plot(xaxis, regression)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(title)
    plt.savefig(filename, dpi=300)

    #slope, r2 = rsquared(np.log(xaxis), np.log(yaxis))
    #print("The exponent is", slope)
    #print("R^2 =", r2)
