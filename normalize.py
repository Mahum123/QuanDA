import numba
from numba import cuda
import cupy as cp
import numpy as np



@cuda.jit
def unnorm_parallel(inp, mean, std, out):
    """out=(inp*std)+mean"""
    x = cuda.grid(1)
    if x < inp.shape[0]:
        out[x] = mean+(std*inp[x])





def inv_norm(inp,mean,std,device):
    if device!=None:
        cuda.select_device(device)

    num_of_samples = inp.shape[0]
    threads_per_block = 1024
    blocks = int(np.ceil((num_of_samples)/1024))
    result_gpu = cp.zeros((num_of_samples), dtype='float64') 
    unnorm_parallel[blocks, threads_per_block](inp,mean,std,result_gpu)

    return result_gpu
