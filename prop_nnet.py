import numba
from numba import cuda
import cupy as cp
import numpy as np
from itertools import product



@cuda.jit
def prop2(perm, probab_inp, bounds, probab_out):
    """out[0] is not maximal"""
    x = cuda.grid(1)
    if x < perm.shape[0]:

        #LB_0 < UB_0 < LB_x < UB_x
        if (bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[1,perm[x][1],0] < bounds[1,perm[x][1],1]) or (bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[2,perm[x][2],0] < bounds[2,perm[x][2],1]) or (bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[3,perm[x][3],0] < bounds[3,perm[x][3],1]) or (bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[4,perm[x][4],0] < bounds[4,perm[x][4],1]):
            probab_out[x] = probab_inp[0,perm[x][0]] * probab_inp[1,perm[x][1]] * probab_inp[2,perm[x][2]] * probab_inp[3,perm[x][3]] * probab_inp[4,perm[x][4]]

        #LB_0 < LB_x < UB_0 < UB_x
        elif (bounds[0,perm[x][0],0] < bounds[1,perm[x][1],0] < bounds[0,perm[x][0],1] < bounds[1,perm[x][1],1]) or (bounds[0,perm[x][0],0] < bounds[2,perm[x][2],0] < bounds[0,perm[x][0],1] < bounds[2,perm[x][2],1]) or (bounds[0,perm[x][0],0] < bounds[3,perm[x][3],0] < bounds[0,perm[x][0],1] < bounds[3,perm[x][3],1]) or (bounds[0,perm[x][0],0] < bounds[4,perm[x][4],0] < bounds[0,perm[x][0],1] < bounds[4,perm[x][4],1]):
            temp = 0
            for i in range(3): #check which node overlaps out[0]
                if (bounds[i+1,perm[x][i+1],0] < bounds[0,perm[x][0],0] < bounds[i+1,perm[x][i+1],1] < bounds[0,perm[x][0],1]):
                    a = probab_inp[1,perm[x][1]]
                    b = probab_inp[2,perm[x][2]]
                    c = probab_inp[3,perm[x][3]]
                    d = probab_inp[4,perm[x][4]]
                    portion = abs(bounds[i+1,perm[x][i+1],1] - bounds[0,perm[x][0],1]) / abs(bounds[i+1,perm[x][i+1],1] - bounds[i+1,perm[x][i+1],0])
                    if i==0:
                        a = a*portion
                    elif i==1:
                        b = b*portion
                    elif i==2:
                        c = c*portion
                    elif i==3:
                        d = d*portion

                    temp = probab_inp[0,perm[x][0]] * a * b * c * d
            probab_out[x] = temp
            
        #LB_x < LB_0 < UB_0 < UB_x
        elif (bounds[1,perm[x][1],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[1,perm[x][1],1]) or (bounds[2,perm[x][2],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[2,perm[x][2],1]) or (bounds[3,perm[x][3],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[3,perm[x][3],1]) or (bounds[4,perm[x][4],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[4,perm[x][4],1]):
            temp = 0
            for i in range(3): #check which node overlaps out[0]
                if (bounds[i+1,perm[x][i+1],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[i+1,perm[x][i+1],1]):
                    a = probab_inp[1,perm[x][1]]
                    b = probab_inp[2,perm[x][2]]
                    c = probab_inp[3,perm[x][3]]
                    d = probab_inp[4,perm[x][4]]
                    portion = abs(bounds[i+1,perm[x][i+1],1] - bounds[0,perm[x][0],1]) / abs(bounds[i+1,perm[x][i+1],1] - bounds[i+1,perm[x][i+1],0])
                    if i==0:
                        a = a*portion
                    elif i==1:
                        b = b*portion
                    elif i==2:
                        c = c*portion
                    elif i==3:
                        d = d*portion

                    temp = probab_inp[0,perm[x][0]] * a * b * c * d
            probab_out[x] = temp


@cuda.jit
def prop4(perm, probab_inp, bounds, probab_out):
    """out[0] is not minimal"""
    x = cuda.grid(1)
    if x < len(perm):

        #LB_x < UB_x < LB_0 < UB_0
        if (bounds[1,perm[x][1],0] < bounds[1,perm[x][1],1] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1]) or (bounds[2,perm[x][2],0] < bounds[2,perm[x][2],1] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1]) or (bounds[3,perm[x][3],0] < bounds[3,perm[x][3],1] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1]) or (bounds[4,perm[x][4],0] < bounds[4,perm[x][4],1] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1]):
            probab_out[x] = probab_inp[0,perm[x][0]] * probab_inp[1,perm[x][1]] * probab_inp[2,perm[x][2]] * probab_inp[3,perm[x][3]] * probab_inp[4,perm[x][4]]

        #LB_x < LB_0 < UB_x < UB_0
        elif (bounds[1,perm[x][1],0] < bounds[0,perm[x][0],0] < bounds[1,perm[x][1],1] < bounds[0,perm[x][0],1]) or (bounds[2,perm[x][2],0] < bounds[0,perm[x][0],0] < bounds[2,perm[x][2],1] < bounds[0,perm[x][0],1]) or (bounds[3,perm[x][3],0] < bounds[0,perm[x][0],0] < bounds[3,perm[x][3],1] < bounds[0,perm[x][0],1]) or (bounds[4,perm[x][4],0] < bounds[0,perm[x][0],0] < bounds[4,perm[x][4],1] < bounds[0,perm[x][0],1]):
            temp = 0
            for i in range(3): #check which node overlaps out[0]
                if (bounds[i+1,perm[x][i+1],0] < bounds[0,perm[x][0],0] < bounds[i+1,perm[x][i+1],1] < bounds[0,perm[x][0],1]):
                    a = probab_inp[1,perm[x][1]]
                    b = probab_inp[2,perm[x][2]]
                    c = probab_inp[3,perm[x][3]]
                    d = probab_inp[4,perm[x][4]]
                    portion = abs(bounds[0,perm[x][0],0] - bounds[i+1,perm[x][i+1],0]) / abs(bounds[i+1,perm[x][i+1],1] - bounds[i+1,perm[x][i+1],0])
                    if i==0:
                        a = a*portion
                    elif i==1:
                        b = b*portion
                    elif i==2:
                        c = c*portion
                    elif i==3:
                        d = d*portion

                    temp = probab_inp[0,perm[x][0]] * a * b * c * d
            probab_out[x] = temp
            
        #LB_x < LB_0 < UB_0 < UB_x
        elif (bounds[1,perm[x][1],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[1,perm[x][1],1]) or (bounds[2,perm[x][2],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[2,perm[x][2],1]) or (bounds[3,perm[x][3],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[3,perm[x][3],1]) or (bounds[4,perm[x][4],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[4,perm[x][4],1]):
            temp = 0
            for i in range(3): #check which node overlaps out[0]
                if (bounds[i+1,perm[x][i+1],0] < bounds[0,perm[x][0],0] < bounds[0,perm[x][0],1] < bounds[i+1,perm[x][i+1],1]):
                    a = probab_inp[1,perm[x][1]]
                    b = probab_inp[2,perm[x][2]]
                    c = probab_inp[3,perm[x][3]]
                    d = probab_inp[4,perm[x][4]]
                    portion = abs(bounds[0,perm[x][0],0] - bounds[i+1,perm[x][i+1],0]) / abs(bounds[i+1,perm[x][i+1],1] - bounds[i+1,perm[x][i+1],0])
                    if i==0:
                        a = a*portion
                    elif i==1:
                        b = b*portion
                    elif i==2:
                        c = c*portion
                    elif i==3:
                        d = d*portion

                    temp = probab_inp[0,perm[x][0]] * a * b * c * d
            probab_out[x] = temp
            





def property(probab,bounds,prop,device):
    if device!=None:
        cuda.select_device(device)

    num_of_output_nodes = 5 #Also: probab.shape[0]
    num_of_segments = probab.shape[1]

    if prop==1: #Property 1: requires only node 1
        result_gpu = 0 #not gpu variable...name for consistency
        for seg in range(num_of_segments):
            if bounds[0,seg,1] <= 1500:
                result_gpu += probab[0,seg]
            else:
                break

    else:  #Property 2-3: comparing results of all output nodes 
        seg = list(range(num_of_segments))
        permutations = np.array(list(product(seg, seg, seg, seg, seg))) #for 5 output nodes
        permutations_gpu = cuda.to_device(permutations)

        threads_per_block = 1024
        blocks = int(np.ceil((len(permutations))/1024))
        result_gpu_array = cp.zeros((len(permutations)), dtype='float64') 

##-----------SANITY CHECK
#        test_probab = 0
#        for x in range(len(permutations)):
#            test_probab += probab[0][permutations[x][0]] * probab[1][permutations[x][1]] * probab[2][permutations[x][2]] * probab[3][permutations[x][3]] * probab[4][permutations[x][4]]
#        print(test_probab) #should sum to 1
##-----------SANITY CHECK

        if prop==2:
            prop2[blocks, threads_per_block](permutations_gpu,probab,bounds,result_gpu_array)
        elif prop==4:
            prop4[blocks, threads_per_block](permutations_gpu,probab,bounds,result_gpu_array)

        result_gpu = cp.sum(result_gpu_array, axis=0)

    return result_gpu
