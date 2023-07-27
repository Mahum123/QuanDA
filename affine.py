import numba
from numba import cuda
import cupy as cp
import numpy as np


@cuda.jit
def mult_parallel(w, inp, out):
    """out[sample,node]=inp[sample,:]*w[:,node]"""
    x, y = cuda.grid(2)
    if x < out.shape[0] and y < out.shape[1]:
        for i in range(inp.shape[1]):
            out[x,y] += inp[x,i]*w[y,i]

@cuda.jit
def add_parallel(b, inp, out):
    """out[sample,node]=inp[sample,node]*b[node]"""
    x, y = cuda.grid(2)
    if x < out.shape[0] and y < out.shape[1]:
        out[x,y] = inp[x,y]+b[y]

@cuda.jit
def add_group_parallel(b, groups, inp, out):
    """For each output node, add inputs from all corresponding input groups and bias (in parallel)"""
    x = cuda.grid(1)
    if x < inp.shape[0]:
        out[x] = b
        for g in range(groups):
            out[x] += inp[x,g]

@cuda.jit
def relu_parallel(inp, out):
    """out=max(inp,0)"""
    x, y = cuda.grid(2)
    if x < out.shape[0] and y < out.shape[1]:
        if inp[x,y]<0:
            out[x,y] = 0
        else:
            out[x,y] = inp[x,y]

@cuda.jit
def relu_group_parallel(inp, out):
    """out=max(inp,0)"""
    x = cuda.grid(1)
    if x < inp.shape[0]:
        if inp[x]<0:
            out[x] = 0
        else:
            out[x] = inp[x]




def mult(inp,w,num_of_segements,device):
    if device!=None:
        cuda.select_device(device)

    num_of_samples = inp.shape[0]
    num_of_input_nodes = inp.shape[1]
    num_of_output_nodes = w.shape[0]
    result_gpu = cp.zeros((num_of_samples,num_of_output_nodes), dtype='float64') 
    block_dimx = int(cp.ceil((num_of_segements**num_of_input_nodes)/32))
    block_dimy = int(cp.ceil(num_of_output_nodes/32))
    mult_parallel[(block_dimx,block_dimy),(32,32)](w,inp,result_gpu)

#    print("num_of_samples: ", num_of_samples)
#    print("num_of_input_nodes: ", num_of_input_nodes)
#    print("num_of_output_nodes: ",num_of_output_nodes)

    return result_gpu

def ad(inp,b,device):
    if device!=None:
        cuda.select_device(device)

    num_of_samples = inp.shape[0]
    num_of_nodes = inp.shape[1]
    result_gpu = cp.zeros((num_of_samples,num_of_nodes), dtype='float64') 
    block_dimx = int(cp.ceil((num_of_samples)/32))
    block_dimy = int(cp.ceil(num_of_nodes/32))
    add_parallel[(block_dimx,block_dimy),(32,32)](b,inp,result_gpu)

    return result_gpu

def ad_group(inp,b,device):
    if device!=None:
        cuda.select_device(device)

    num_of_samples = inp.shape[0]
    num_of_groups = inp.shape[1]
    threads_per_block = 1024
    blocks = int(np.ceil((num_of_samples)/1024))
    result_gpu = cp.zeros((num_of_samples), dtype='float64') 
    add_group_parallel[blocks, threads_per_block](b,num_of_groups,inp,result_gpu)

    return result_gpu

def relu(inp,device):
    if device!=None:
        cuda.select_device(device)

    num_of_samples = inp.shape[0]
    num_of_nodes = inp.shape[1]
    result_gpu = cp.zeros((num_of_samples,num_of_nodes), dtype='float64') 
    block_dimx = int(cp.ceil((num_of_samples)/32))
    block_dimy = int(cp.ceil(num_of_nodes/32))
    relu_parallel[(block_dimx,block_dimy),(32,32)](inp,result_gpu)

    return result_gpu

def relu_group(inp,device):
    if device!=None:
        cuda.select_device(device)

    num_of_samples = inp.shape[0]
    threads_per_block = 1024
    blocks = int(np.ceil((num_of_samples)/1024))
    result_gpu = cp.zeros((num_of_samples), dtype='float64') 
    relu_group_parallel[blocks, threads_per_block](inp,result_gpu)

    return result_gpu





if __name__ == '__main__':
    print("Run main.py")
