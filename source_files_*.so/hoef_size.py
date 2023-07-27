import numpy as np


def hoef_size(delta,eps):
    #By Hoeffding's Inequality:	ite > ln(2/delta)/2*eps^2
    ite = np.ceil(np.log(2/round(1-delta,3)) / (2*eps**2))
    return ite


if __name__ == '__main__':
    print("Run main.py")

