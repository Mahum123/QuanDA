from numba import cuda
import numba
import numpy as np
import cupy as cp
import os
import sys
import shutil
import itertools
import decimal

import network_parameters_nnet as npara
import input_parameters_nnet as ipara
import constrained_rand as generator
import affine as nn
import counter as ct
np.set_printoptions(threshold=sys.maxsize)


def estimate_probab(nnet,prop,l,seg,fileNameInput,fileNameOutput,number_of_nodes_input,number_of_nodes_output,group_index,total_number_of_groups,device):

    if device!=None:
        cuda.select_device(device)

    os.makedirs(os.path.dirname("Datasets/ACAS_Xu/Probablity_Estimates/"), exist_ok=True)
    #FOR LOADING INPUT AND OUTPUT
    if fileNameInput==None:
        if prop==0 or prop==999:
            fileNameInput = "Datasets/ACAS_Xu/Input_Bounds/layer0_complete.npy"
            fileNameOutput = "Datasets/ACAS_Xu/Computed_Output_Bounds/layer"+str(l+1)+"_complete.npy"
            fileNameProbab = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_complete.npy"
        elif prop==1:
            fileNameInput = "Datasets/ACAS_Xu/Input_Bounds/layer0_prop1.npy"
            fileNameOutput = "Datasets/ACAS_Xu/Computed_Output_Bounds/layer"+str(l+1)+"_prop"+str(prop)+".npy"
            fileNameProbab = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_prop"+str(prop)+".npy"
        elif prop==2:
            fileNameInput = "Datasets/ACAS_Xu/Input_Bounds/layer0_prop2.npy"
            fileNameOutput = "Datasets/ACAS_Xu/Computed_Output_Bounds/layer"+str(l+1)+"_prop"+str(prop)+".npy"
            fileNameProbab = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_prop"+str(prop)+".npy"
        elif prop==3:
            fileNameInput = "Datasets/ACAS_Xu/Input_Bounds/layer0_prop3.npy"
            fileNameOutput = "Datasets/ACAS_Xu/Computed_Output_Bounds/layer"+str(l+1)+"_prop"+str(prop)+".npy"
            fileNameProbab = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_prop"+str(prop)+".npy"
        elif prop==4:
            fileNameInput = "Datasets/ACAS_Xu/Input_Bounds/layer0_prop4.npy"
            fileNameOutput = "Datasets/ACAS_Xu/Computed_Output_Bounds/layer"+str(l+1)+"_prop"+str(prop)+".npy"
            fileNameProbab = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_prop"+str(prop)+".npy"

        inp = np.transpose(np.load(fileNameInput))
        out = np.load(fileNameOutput)

    else:
        inp = np.load(fileNameInput)
        out = np.load(fileNameOutput)

        if prop==0 or prop==999:
            fileNameProbab = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_group"+str(group_index+1)+"_complete.npy"
        else:
            fileNameProbab = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_prop"+str(prop)+"_group"+str(group_index+1)+".npy"

#    print("Input: ",len(inp[0]),"\nOutput: ", len(out[0]))
#    print("Number of input node: ",number_of_nodes_input,"\nNumber of output node: ",number_of_nodes_output)
#    print("\n\nInput file: ", fileNameInput,"\nOutput File:", fileNameOutput,"\nProbability File: ", fileNameProbab)

    #IMPORTING NETWORK AND INPUT PARAMETERS
    w1, w2, w3, w4, w5, w6, w7, b1, b2, b3, b4, b5, b6, b7 = npara.parameters(open(nnet, "r"))
    mean, std_dev, lb, ub = ipara.parameters(open(nnet, "r"))

    if l == 0:
        w = np.array(w1)
        b = np.array(b1)
        w_gpu = cuda.to_device(w1) # move data to the device
        b_gpu = cuda.to_device(b1) # move data to the device
    elif l==1:
        w = np.array(w2)
        b = np.array(b2)
        w_gpu = cuda.to_device(w2) # move data to the device
        b_gpu = cuda.to_device(b2) # move data to the device
    elif l==2:
        w = np.array(w3)
        b = np.array(b3)
        w_gpu = cuda.to_device(w3) # move data to the device
        b_gpu = cuda.to_device(b3) # move data to the device
    elif l==3:
        w = np.array(w4)
        b = np.array(b4)
        w_gpu = cuda.to_device(w4) # move data to the device
        b_gpu = cuda.to_device(b4) # move data to the device
    elif l==4:
        w = np.array(w5)
        b = np.array(b5)
        w_gpu = cuda.to_device(w5) # move data to the device
        b_gpu = cuda.to_device(b5) # move data to the device
    elif l==5:
        w = np.array(w6)
        b = np.array(b6)
        w_gpu = cuda.to_device(w6) # move data to the device
        b_gpu = cuda.to_device(b6) # move data to the device
    elif l==6:
        w = np.array(w7)
        b = np.array(b7)
        w_gpu = cuda.to_device(w7) # move data to the device
        b_gpu = cuda.to_device(b7) # move data to the device

    inp_mean_np = cp.array(mean[0][0:len(mean[0])-1])
    inp_std_dev_np = cp.array(std_dev[0][0:len(std_dev[0])-1])
    out_mean_np = mean[0][len(mean[0])-1]
    out_std_dev_np = std_dev[0][len(std_dev[0])-1]

    #IDENTIFY INPUT AND OUTPUT SEGMENTS
    input_seg = np.zeros((number_of_nodes_input,seg,2))
    output_seg = np.zeros((number_of_nodes_output,seg,2))

    for n in range(number_of_nodes_input):
        for s in range(seg):
            input_seg[n][s][0] = inp[0][(group_index*number_of_nodes_input)+n] + (s*((inp[1][(group_index*number_of_nodes_input)+n]-inp[0][(group_index*number_of_nodes_input)+n])/seg))
            input_seg[n][s][1] = inp[0][(group_index*number_of_nodes_input)+n] + ((s+1)*((inp[1][(group_index*number_of_nodes_input)+n]-inp[0][(group_index*number_of_nodes_input)+n])/seg))

    for n in range(number_of_nodes_output):
        if out[0][n]==0 and out[1][n]==0:
            output_seg[n][0][0] = 0
            output_seg[n][0][1] = 0
            for s in range(1,seg):
                output_seg[n][s][0] = float('NaN')
                output_seg[n][s][1] = float('NaN')
        else:
            for s in range(seg):
                output_seg[n][s][0] = out[0][n] + (s*((out[1][n]-out[0][n])/seg))
                output_seg[n][s][1] = out[0][n] + ((s+1)*((out[1][n]-out[0][n])/seg))

    #ESTIMATING PROBABILITY 
    if numba.cuda.is_available(): 

        #==FOR LAYER 1:
        if l == 0: #input layer
            sample_inp = generator.samples(number_of_nodes_input,seg,input_seg,device)
            sample_inp = (sample_inp - inp_mean_np)/inp_std_dev_np
            sample_out = nn.mult(sample_inp,w_gpu,seg,device)
            sample_out = nn.ad(sample_out,b_gpu,device)
            sample_out = nn.relu(sample_out,device)

            #CHECK WHICH OUTPUT SEGMENT SAMPLE_OUT BELONGS TO
            node_freq = ct.output_segment(sample_out,output_seg,device) 
#            print(cp.sum(node_freq/sample_out.shape[0],axis=1)) #sanity check: sum of probabs 1 for each node
            probab = node_freq/sample_out.shape[0]


        #==FOR LAYER 2-6:
        else:

            #FOR ESTIMATES WITHIN INDIVIDUAL GROUPS
            if group_index < total_number_of_groups:
                sample_inp = generator.samples(number_of_nodes_input,seg,input_seg,device)
                sample_out = nn.mult(sample_inp,w_gpu[:,number_of_nodes_input*group_index:number_of_nodes_input*(group_index+1)],seg,device)


                #CHECK WHICH OUTPUT SEGMENT SAMPLE_OUT BELONGS TO
                if prop==0 or prop==999:
                    fileNameProbabInp = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l)+"_complete.npy"
                else:
                    fileNameProbabInp = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l)+"_prop"+str(prop)+".npy"
                inp_probab = np.load(fileNameProbabInp)
                if inp_probab.ndim == 3:
                    inp_probab = inp_probab[:,:,inp_probab.shape[2]-1].copy() #probab from current iteration
                inp_probab_gpu = cuda.to_device(inp_probab)

                probab = ct.output_segment_weighted(sample_out,output_seg,inp_probab_gpu,group_index,number_of_nodes_input,device) 
                total = cp.sum(probab, axis=1)
                probab = probab/total[:,None]
            else:
                raise IndexError("System encountered an INVALID group_index. Report Issue.")


    else:
        raise NotImplementedError("CUDA not detected! \nFramework does not support CPU implementation.")


    #SAVE PROBABILITY ESTIMATE IN NEW FILE
    if os.path.isfile(fileNameProbab) == False:
        np.save(fileNameProbab, probab) #if file does not exist, save estimate
    else:
        if l==0:
            old_estimate = np.load(fileNameProbab)
            updated_estimate = np.dstack((old_estimate,probab))
#            print(updated_estimate) #size: [50 x 5 x i]
            np.save(fileNameProbab,updated_estimate) #if file exists, update 
        else:
            np.save(fileNameProbab, probab) #overwrite group probabilities (only retain final layer probab for all iterations)



def group_bounds(nnet,prop,l,seg,total_number_of_groups,number_of_nodes_input,comp_flag,device):

    os.makedirs(os.path.dirname("Datasets/ACAS_Xu/Computed_Output_Bounds_Groups/"), exist_ok=True)

    #IMPORTING NETWORK PARAMETERS
    w1, w2, w3, w4, w5, w6, w7, b1, b2, b3, b4, b5, b6, b7 = npara.parameters(open(nnet, "r"))
#    mean, std_dev, lb, ub = ipara.parameters(open(nnet, "r")) #l!=0 --> No input normalization required

    if l==1:
        w = np.array(w2)
        b = np.array(b2)
    elif l==2:
        w = np.array(w3)
        b = np.array(b3)
    elif l==3:
        w = np.array(w4)
        b = np.array(b4)
    elif l==4:
        w = np.array(w5)
        b = np.array(b5)
    elif l==5:
        w = np.array(w6)
        b = np.array(b6)
    elif l==6:
        w = np.array(w7)
        b = np.array(b7)


    #NODES PER GROUP:
    nodes_per_group = number_of_nodes_input//total_number_of_groups

    #IMPORT INPUT BOUNDS
    if prop==0 or prop==999:
        fileNameInput = "Datasets/ACAS_Xu/Computed_Output_Bounds/layer"+str(l)+"_complete.npy"
    else:
        fileNameInput = "Datasets/ACAS_Xu/Computed_Output_Bounds/layer"+str(l)+"_prop"+str(prop)+".npy"
    inp = np.load(fileNameInput)
    lb = np.transpose(inp[0])
    ub = np.transpose(inp[1])

    #FOR LAYER 1: 
    if l == 0: #input layer
        raise ValueError("ACAS Xu input laer has only 5 nodes, and hence should not be divided into groups")

    #FOR LAYER 2-6:
    else: 
        number_of_nodes_output = len(b)
        for g in range(total_number_of_groups):
            if prop==0 or prop==999:
                fileNameOutput = "Datasets/ACAS_Xu/Computed_Output_Bounds_Groups/layer"+str(l+1)+"_group"+str(g+1)+"_complete.npy"
            else:
                fileNameOutput = "Datasets/ACAS_Xu/Computed_Output_Bounds_Groups/layer"+str(l+1)+"_prop"+str(prop)+"_group"+str(g+1)+".npy"


            #COMPUTE GROUP BOUNDS
            if comp_flag:
                n1_extremum = np.vstack((lb[nodes_per_group*g:nodes_per_group*(g+1)], ub[nodes_per_group*g:nodes_per_group*(g+1)]))
                n2_max = [0] * len(b)
                for i in range(len(b)): #for each output neuron
                    for j in range(nodes_per_group*g,nodes_per_group*(g+1)): #for each input neuron in group
                    #parameter columns 10g : 10(g+1) --> where 10=nodes_per_group
                        if w[i][j] <= 0:
                            n2_max[i] = n2_max[i] + (w[i][j]*lb[j])
                        else:
                            n2_max[i] = n2_max[i] + (w[i][j]*ub[j])

                n2_min = [0] * len(b)
                for i in range(len(b)): #for each output neuron
                    for j in range(nodes_per_group*g,nodes_per_group*(g+1)): #for each input neuron in group
                    #parameter columns 10g : 10(g+1) --> where 10=nodes_per_group
                        if w[i][j] <= 0:
                            n2_min[i] = n2_min[i] + (w[i][j]*ub[j])
                        else:
                            n2_min[i] = n2_min[i] + (w[i][j]*lb[j])

                n2_extremum = np.vstack((n2_min, n2_max))
                np.save(fileNameOutput,n2_extremum)

            #ESTIMATE PROBABILITY FOR EACH GROUP
            estimate_probab(nnet,prop,l,seg,fileNameInput,fileNameOutput,nodes_per_group,number_of_nodes_output,g,total_number_of_groups,device)




if __name__ == '__main__':
    print("Run main.py")
