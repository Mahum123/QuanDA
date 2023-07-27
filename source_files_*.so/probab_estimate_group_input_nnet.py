from numba import cuda
import numba
import numpy as np
import cupy as cp
import os
import sys
import itertools
import decimal

import network_parameters_nnet as npara
import output_parameters_nnet as opara
import constrained_rand as generator
import affine as nn
import normalize as norm
import counter as ct
import prop_nnet as propx
np.set_printoptions(threshold=sys.maxsize)


def estimate_probab(nnet,prop,l,seg,number_of_nodes_output,total_number_of_groups,device):

    if device!=None:
        cuda.select_device(device)

    number_of_nodes_input = total_number_of_groups


    #COMPUTE BOUNDS FOR EACH OUTPUT NODE (ACC. TO GROUP ESTIMATE AS INPUT)
    input_seg = np.zeros((number_of_nodes_output,number_of_nodes_input,seg,2))
    output_seg = np.zeros((number_of_nodes_output,seg,2))

    #For inp_seg
    for g in range(total_number_of_groups):
        if prop==0 or prop==999:
            fileNameInput = "Datasets/ACAS_Xu/Computed_Output_Bounds_Groups/layer"+str(l+1)+"_group"+str(g+1)+"_complete.npy"
        else:
            fileNameInput = "Datasets/ACAS_Xu/Computed_Output_Bounds_Groups/layer"+str(l+1)+"_prop"+str(prop)+"_group"+str(g+1)+".npy"
        bounds_inp = np.load(fileNameInput)

        for n in range(number_of_nodes_output):
            for s in range(seg):
                input_seg[n][g][s][0] = bounds_inp[0][n] + (s*((bounds_inp[1][n]-bounds_inp[0][n])/seg))
                input_seg[n][g][s][1] = bounds_inp[0][n] + ((s+1)*((bounds_inp[1][n]-bounds_inp[0][n])/seg))

    #For output_seg
    if prop==0 or prop==999:
        fileNameOutput = "Datasets/ACAS_Xu/Computed_Output_Bounds/layer"+str(l+1)+"_complete.npy"
        fileNameProbab = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_complete.npy"
    else:
        fileNameOutput = "Datasets/ACAS_Xu/Computed_Output_Bounds/layer"+str(l+1)+"_prop"+str(prop)+".npy"
        fileNameProbab = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_prop"+str(prop)+".npy"
    out = np.load(fileNameOutput)

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


    #IMPORTING NETWORK AND INPUT PARAMETERS
    w1, w2, w3, w4, w5, w6, w7, b1, b2, b3, b4, b5, b6, b7 = npara.parameters(open(nnet, "r"))

    if l==1:
        b = np.array(b2)
        b_gpu = cuda.to_device(b2) # move data to the device
    elif l==2:
        b = np.array(b3)
        b_gpu = cuda.to_device(b3) # move data to the device
    elif l==3:
        b = np.array(b4)
        b_gpu = cuda.to_device(b4) # move data to the device
    elif l==4:
        b = np.array(b5)
        b_gpu = cuda.to_device(b5) # move data to the device
    elif l==5:
        b = np.array(b6)
        b_gpu = cuda.to_device(b6) # move data to the device
    elif l==6:
        b = np.array(b7)
        b_gpu = cuda.to_device(b7) # move data to the device


    #GENERATE ARRAYS FOR PROBABILITY OF INPUTS (FROM GROUPS)
    probab_inp = np.zeros((number_of_nodes_output,number_of_nodes_input,seg))
    for g in range(total_number_of_groups):
        if prop==0 or prop==999:
            fileNameProbabInp = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_group"+str(g+1)+"_complete.npy"
        else:
            fileNameProbabInp = "Datasets/ACAS_Xu/Probablity_Estimates/layer"+str(l+1)+"_prop"+str(prop)+"_group"+str(g+1)+".npy"
        if os.path.isfile(fileNameProbabInp):
            probab_inp[:,g,:] = np.load(fileNameProbabInp)
            os.remove(fileNameProbabInp)
        else:
            raise FileNotFoundError(str(fileNameProbabInp)+" not found. Run probab_estimate_nnet.py.")



    #ESTIMATE PROBABILITY FOR LAYER
    if numba.cuda.is_available(): 
        probab = cp.zeros((number_of_nodes_output,seg))

        for n in range(number_of_nodes_output):
            sample_inp = generator.samples(number_of_nodes_input,seg,input_seg[n,:,:,:],device)
            sample_out = nn.ad_group(sample_inp,b_gpu[n],device)

            if l==6: #unnormalize
                mean, std = opara.out_para(nnet)
                sample_out = norm.inv_norm(sample_out,mean,std,device)
#                print("Shape of normalized output: ",sample_out.shape)
            else:
                sample_out = nn.relu_group(sample_out,device)

            sample_out_cpu = cuda.to_device(sample_out).copy_to_host()
            sample_out_cpu = sample_out_cpu.reshape((1,sample_out_cpu.shape[0]))
            sample_out = cuda.to_device(sample_out_cpu)
#            print(output_seg[n,:,:].reshape((1, output_seg[n,:,:].shape[0], output_seg[n,:,:].shape[1])))
#            print("Going tp the counter: ",sample_out.shape)

            probab[n,:] = ct.output_segment_weighted(cp.transpose(sample_out),output_seg[n,:,:].reshape((1, output_seg[n,:,:].shape[0], output_seg[n,:,:].shape[1])),probab_inp[n,:,:],None,number_of_nodes_input,device)

        total = cp.sum(probab, axis=1)
        probab = probab/total[:,None]

        if l==6:
            if prop!=0:
                probab = propx.property(probab,output_seg,prop,device) 
#                print(probab)
                probab = np.atleast_1d(probab)

    else:
        raise NotImplementedError("CUDA not detected! \nFramework does not support CPU implementation.")




    #SAVE PROBABILITY ESTIMATE IN NEW FILE
    if os.path.isfile(fileNameProbab) == False:
        np.save(fileNameProbab, probab) #if file does not exist, save estimate
    else:
        old_estimate = np.load(fileNameProbab)
        updated_estimate = np.dstack((old_estimate,probab))
        np.save(fileNameProbab,updated_estimate) #if file exists, update 


    return fileNameProbab, output_seg



if __name__ == '__main__':
    print("Run main.py")

