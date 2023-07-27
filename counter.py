from numba import cuda
import numba
import numpy as np
import cupy as cp
#import math

@cuda.jit
def counter_parallel_singlenode(n, inp, bounds, freq):
    """For each input and node (in parallel), check which segment of node input belongs to"""
    x = cuda.grid(1)
    if x < inp.shape[0]:
        for j in range(bounds.shape[1]): #checking all segments
            if inp[x,n] >= bounds[n,j,0] and inp[x,n] <= bounds[n,j,1]:
                freq[x,j] += 1 #(inputs,num_of_segements)



@cuda.jit
def weighted_counter_parallel_singlenode(n, g, I, num_seg, out, bounds_out, probab, freq):
    """For each input and node (in parallel), check which segment of node input belongs to"""
    x = cuda.grid(1)
    if x < out.shape[0]:
        for j in range(bounds_out.shape[1]): #checking all output segments
            if out[x,n] >= bounds_out[n,j,0] and out[x,n] <= bounds_out[n,j,1]:
                for i in range(I): #because number of segments is consistent in all layers
                    seg = x//(num_seg**i)%num_seg 
                    freq[x,j] += probab[i+(I*g),seg]





def output_segment(samples_gpu,bounds,device):
    if device!=None:
        cuda.select_device(device)

    number_of_inputs = samples_gpu.shape[0]
    num_of_nodes_input = samples_gpu.shape[1]
    num_of_nodes_output = bounds.shape[0]
    num_of_segements = bounds.shape[1]
#    print("number_of_inputs: ",number_of_inputs,"    num_of_nodes_input: ",num_of_nodes_input,"    num_of_segements: ",num_of_segements)
    bounds_gpu = cuda.to_device(bounds)
    freq_final = cp.zeros((num_of_nodes_output,num_of_segements), dtype='int64') #(number_of_nodes,num_of_segements)
    threads_per_block = 1024
    blocks = int(np.ceil((number_of_inputs)/1024))

    #For GPU memory management: break problem into sub-problems
    for n in range(num_of_nodes_output):
        freq_gpu = cp.zeros((number_of_inputs,num_of_segements), dtype='int64') #(inputs,num_of_segements)
        counter_parallel_singlenode[blocks, threads_per_block](n, samples_gpu, bounds_gpu, freq_gpu)
        freq_final[n] = cp.sum(freq_gpu, axis=0)
#        freq_final += cp.sum(freq_gpu, axis=0)
        freq_gpu = None

    return freq_final



def output_segment_weighted(samples_out_gpu,bounds_out,inp_probab_gpu,group_index,num_of_nodes_input,device):
    if device!=None:
        cuda.select_device(device)

    number_of_inputs = samples_out_gpu.shape[0]
    if group_index==None:
        num_of_nodes_output = 1
    else:
        num_of_nodes_output = samples_out_gpu.shape[1] #or: bounds_out.shape[0]
    num_of_segements = bounds_out.shape[1]
    bounds_out_gpu = cuda.to_device(bounds_out)

    freq_final = cp.zeros((num_of_nodes_output,num_of_segements), dtype='float64') #(number_of_nodes,num_of_segements)
    threads_per_block = 1024
    blocks = int(np.ceil((number_of_inputs)/1024))

    #FOR GROUPS OF INPUT NODES TO OUTPUT NODES
    if group_index!=None:
        #For GPU memory management: break problem into sub-problems
        for n in range(num_of_nodes_output):
            freq_gpu = cp.zeros((number_of_inputs,num_of_segements), dtype='float64') #(inputs,num_of_segements)
            weighted_counter_parallel_singlenode[blocks, threads_per_block](n, group_index, num_of_nodes_input, num_of_segements, samples_out_gpu, bounds_out_gpu, inp_probab_gpu, freq_gpu)
            freq_final[n] = cp.sum(freq_gpu, axis=0)
            freq_gpu = None

    #FROM GROUPS TO COMPLETE OUTPUT NODE ESTIMATES
    else:
        freq_gpu = cp.zeros((number_of_inputs,num_of_segements), dtype='float64') #(inputs,num_of_segements)
        weighted_counter_parallel_singlenode[blocks, threads_per_block](0, 0, num_of_nodes_input, num_of_segements, samples_out_gpu, bounds_out_gpu, inp_probab_gpu, freq_gpu)
        freq_final = cp.sum(freq_gpu, axis=0)

#    print("\n",cp.sum(freq_final, axis=1))
    return freq_final



if __name__ == '__main__':
    print("Run main.py")
