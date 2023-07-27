import numba
from numba import cuda
import numpy as np
import cupy as cp
import math


@cuda.jit
def generate_samples(bounds, n, s, inp):
    x, y = cuda.grid(2)
    if x < inp.shape[0] and y < inp.shape[1]:
        rem = x%(s**(y+1))
        seg = int(math.floor(rem/(s**y)))
        lb = bounds[y,seg,0]
        ub = bounds[y,seg,1]
        inp[x,y] = (ub-lb)*inp[x,y] + lb


def samples(number_of_nodes_input,seg,bounds,device):

    if device!=None:
        cuda.select_device(device)

    bounds_gpu = cuda.to_device(bounds)
    inputs = cp.random.random((seg**number_of_nodes_input,number_of_nodes_input), dtype='float64')
#    print(cp.max(cp.ceil(cp.sum(inputs, axis=1))))
    block_dimx = int(np.ceil((seg**number_of_nodes_input)/32))
    block_dimy = int(np.ceil(number_of_nodes_input/32))
#    print("Block Dimensions: ",block_dimx,block_dimy)

    generate_samples[(block_dimx,block_dimy),(32,32)](bounds_gpu, number_of_nodes_input, seg, inputs)
    return inputs


if __name__ == '__main__':
    print("Run main.py")

