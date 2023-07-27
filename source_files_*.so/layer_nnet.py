import numpy as np
import os
import sys
import shutil
import itertools
import network_parameters_nnet as npara
import input_parameters_nnet as ipara


def number_of_nodes(nnet,layer):
    w1, w2, w3, w4, w5, w6, w7, b1, b2, b3, b4, b5, b6, b7 = npara.parameters(open(nnet, "r"))
    if  layer==0:
        return 5, len(b1)
    elif layer==1:
        return len(b1), len(b2)
    elif layer==2:
        return len(b2), len(b3)
    elif layer==3:
        return len(b3), len(b4)
    elif layer==4:
        return len(b4), len(b5)
    elif layer==5:
        return len(b5), len(b6)
    elif layer==6:
        return len(b6), len(b7)
    else: #shouldn't run if number of layers defined correctly!
        return len(b7), 0


def compute_complete_bounds(lb,ub,nnet,prop,ver):
    w1, w2, w3, w4, w5, w6, w7, b1, b2, b3, b4, b5, b6, b7 = npara.parameters(open(nnet, "r"))
    mean, std_dev, lb_orig, ub_orig = ipara.parameters(open(nnet, "r"))

    #Converting arrays to numPy arrays
    w1 = np.array(w1)
    w2 = np.array(w2)
    w3 = np.array(w3)
    w4 = np.array(w4)
    w5 = np.array(w5)
    w6 = np.array(w6)
    w7 = np.array(w7)
    b1 = np.array(b1)
    b2 = np.array(b2)
    b3 = np.array(b3)
    b4 = np.array(b4)
    b5 = np.array(b5)
    b6 = np.array(b6)
    b7 = np.array(b7)

    #Input Normalization
    inp_mean_np = np.array(mean[0][0:len(mean[0])-1])
    inp_std_dev_np = np.array(std_dev[0][0:len(std_dev[0])-1])
    out_mean_np = mean[0][len(mean[0])-1]
    out_std_dev_np = std_dev[0][len(std_dev[0])-1]

    if lb==None and ub==None:
        if prop==0:
            lb = (np.array(lb[0]) - inp_mean_np)/inp_std_dev_np
            ub = (np.array(ub[0]) - inp_mean_np)/inp_std_dev_np
            extremum = np.vstack((lb, ub))
        elif prop==1:
            lb = (np.array([55947.691, -3.141593, -3.141593, 1145.0, 0.0]) - inp_mean_np)/inp_std_dev_np
            ub = (np.array([60760.0, 3.141593, 3.141593, 1200.0, 60.0]) - inp_mean_np)/inp_std_dev_np
            extremum = np.vstack((lb, ub))
        elif prop==2:
            lb = (np.array([55947.691, -3.141593, -3.141593, 1145.0, 0.0]) - inp_mean_np)/inp_std_dev_np
            ub = (np.array([60760.0, 3.141593, 3.141593, 1200.0, 60.0]) - inp_mean_np)/inp_std_dev_np
            extremum = np.vstack((lb, ub))
        elif prop==3:
            lb = (np.array([1500.0, -0.06, 3.1, 980.0, 960.0]) - inp_mean_np)/inp_std_dev_np
            ub = (np.array([1800.0, 0.06, 3.141593, 1200.0, 1200.0]) - inp_mean_np)/inp_std_dev_np
            extremum = np.vstack((lb, ub))
        elif prop==4:
            print	("Here?")
            lb = (np.array([1500.0, -0.06, 0.0, 1000.0, 700.0]) - inp_mean_np)/inp_std_dev_np
            ub = (np.array([1800.0, 0.06, 0.0, 1200.0, 800.0]) - inp_mean_np)/inp_std_dev_np
            extremum = np.vstack((lb, ub))
#        print("Extremum:\n",extremum)
    else:
        print(np.vstack((lb, ub)))
        extremum = np.vstack((lb, ub))
##---Testing Inputs (temp)
#    test_inputs = [[-0.31182839647533234, 0.0, -0.2387324146378273, -0.5, -0.4166666666666667], [-0.16247807039378703, -0.4774648292756546, -0.2387324146378273, -0.3181818181818182, -0.25], [-0.2454504737724233, -0.4774648292756546, 0.0, -0.3181818181818182, 0.0]]
#    extremum = np.vstack((test_inputs[2], test_inputs[2]))
#    print("Input Bounds\n", extremum)
##------------------------

    #LAYER1 BOUNDS
    n1_max = [0] * len(b1)
    for i in range(len(w1)): #for each neuron in layer
        for j in range(len(w1[0])): #for each incoming edge
            if w1[i][j] <= 0:
                n1_max[i] = n1_max[i] + (w1[i][j]*extremum[0][j])
                if ver:
                    print("Layer 1 input for upper bound:", extremum[0][j], "   --node: ", i+1)
            else:
                n1_max[i] = n1_max[i] + (w1[i][j]*extremum[1][j])
                if ver:
                    print("Layer 1 input for upper bound:", extremum[1][j], "   --node: ", i+1)
        n1_max[i] = n1_max[i] + b1[i]
        n1_max[i] = max(0, n1_max[i])
    n1_min = [0] * len(b1)
    for i in range(len(w1)): #for each neuron in layer
        for j in range(len(w1[0])): #for each incoming edge
            if w1[i][j] <= 0:
                n1_min[i] = n1_min[i] + (w1[i][j]*extremum[1][j])
                if ver:
                    print("Layer 1 input for lower bound:", extremum[1][j], "   --node: ", i+1)
            else:
                n1_min[i] = n1_min[i] + (w1[i][j]*extremum[0][j])
                if ver:
                    print("Layer 1 input for lower bound:", extremum[0][j], "   --node: ", i+1)
        n1_min[i] = n1_min[i] + b1[i]
        n1_min[i] = max(0, n1_min[i])
    n1_extremum = np.vstack((n1_min, n1_max))
#    print("Bounds of layer n1\n", n1_extremum, "\n")

    #LAYER2 BOUNDS
    n2_max = [0] * len(b2)
    for i in range(len(w2)): #for each neuron in layer
        for j in range(len(w2[0])): #for each incoming edge
            if w2[i][j] <= 0:
                n2_max[i] = n2_max[i] + (w2[i][j]*n1_extremum[0][j])
                if ver:
                    print("Layer 2 input for upper bound:", n1_extremum[0][j], "   --node: ", i+1)
            else:
                n2_max[i] = n2_max[i] + (w2[i][j]*n1_extremum[1][j])
                if ver:
                    print("Layer 2 input for upper bound:", n1_extremum[1][j], "   --node: ", i+1)
        n2_max[i] = n2_max[i] + b2[i]
        n2_max[i] = max(0, n2_max[i])
    n2_min = [0] * len(b2)
    for i in range(len(w2)): #for each neuron in layer
        for j in range(len(w2[0])): #for each incoming edge
            if w2[i][j] <= 0:
                n2_min[i] = n2_min[i] + (w2[i][j]*n1_extremum[1][j])
                if ver:
                    print("Layer 2 input for lower bound:", n1_extremum[1][j], "   --node: ", i+1)
            else:
                n2_min[i] = n2_min[i] + (w2[i][j]*n1_extremum[0][j])
                if ver:
                    print("Layer 2 input for lower bound:", n1_extremum[0][j], "   --node: ", i+1)
        n2_min[i] = n2_min[i] + b2[i]
        n2_min[i] = max(0, n2_min[i])
    n2_extremum = np.vstack((n2_min, n2_max))
#    print("Bounds of layer n2\n", n2_extremum, "\n")

    #LAYER3 BOUNDS
    n3_max = [0] * len(b3)
    for i in range(len(w3)): #for each neuron in layer
        for j in range(len(w3[0])): #for each incoming edge
            if w3[i][j] <= 0:
                n3_max[i] = n3_max[i] + (w3[i][j]*n2_extremum[0][j])
                if ver:
                    print("Layer 3 input for upper bound:", n2_extremum[0][j], "   --node: ", i+1)
            else:
                n3_max[i] = n3_max[i] + (w3[i][j]*n2_extremum[1][j])
                if ver:
                    print("Layer 3 input for upper bound:", n2_extremum[1][j], "   --node: ", i+1)
        n3_max[i] = n3_max[i] + b3[i]
        n3_max[i] = max(0, n3_max[i])
    n3_min = [0] * len(b3)
    for i in range(len(w3)): #for each neuron in layer
        for j in range(len(w3[0])): #for each incoming edge
            if w3[i][j] <= 0:
                n3_min[i] = n3_min[i] + (w3[i][j]*n2_extremum[1][j])
                if ver:
                    print("Layer 3 input for lower bound:", n2_extremum[1][j], "   --node: ", i+1)
            else:
                n3_min[i] = n3_min[i] + (w3[i][j]*n2_extremum[0][j])
                if ver:
                    print("Layer 3 input for lower bound:", n2_extremum[0][j], "   --node: ", i+1)
        n3_min[i] = n3_min[i] + b3[i]
        n3_min[i] = max(0, n3_min[i])
    n3_extremum = np.vstack((n3_min, n3_max))
#    print("Bounds of layer n3\n", n3_extremum, "\n")

    #LAYER4 BOUNDS
    n4_max = [0] * len(b4)
    for i in range(len(w4)): #for each neuron in layer
        for j in range(len(w4[0])): #for each incoming edge
            if w4[i][j] <= 0:
                n4_max[i] = n4_max[i] + (w4[i][j]*n3_extremum[0][j])
                if ver:
                    print("Layer 4 input for upper bound:", n3_extremum[0][j], "   --node: ", i+1)
            else:
                n4_max[i] = n4_max[i] + (w4[i][j]*n3_extremum[1][j])
                if ver:
                    print("Layer 4 input for upper bound:", n3_extremum[1][j], "   --node: ", i+1)
        n4_max[i] = n4_max[i] + b4[i]
        n4_max[i] = max(0, n4_max[i])
    n4_min = [0] * len(b4)
    for i in range(len(w4)): #for each neuron in layer
        for j in range(len(w4[0])): #for each incoming edge
            if w4[i][j] <= 0:
                n4_min[i] = n4_min[i] + (w4[i][j]*n3_extremum[1][j])
                if ver:
                    print("Layer 4 input for lower bound:", n3_extremum[1][j], "   --node: ", i+1)
            else:
                n4_min[i] = n4_min[i] + (w4[i][j]*n3_extremum[0][j])
                if ver:
                    print("Layer 4 input for lower bound:", n3_extremum[0][j], "   --node: ", i+1)
        n4_min[i] = n4_min[i] + b4[i]
        n4_min[i] = max(0, n4_min[i])
    n4_extremum = np.vstack((n4_min, n4_max))
#    print("Bounds of layer n4\n", n4_extremum, "\n")

    #LAYER5 BOUNDS
    n5_max = [0] * len(b5)
    for i in range(len(w5)): #for each neuron in layer
        for j in range(len(w5[0])): #for each incoming edge
            if w5[i][j] <= 0:
                n5_max[i] = n5_max[i] + (w5[i][j]*n4_extremum[0][j])
                if ver:
                    print("Layer 5 input for upper bound:", n4_extremum[0][j], "   --node: ", i+1)
            else:
                n5_max[i] = n5_max[i] + (w5[i][j]*n4_extremum[1][j])
                if ver:
                    print("Layer 5 input for upper bound:", n4_extremum[1][j], "   --node: ", i+1)
        n5_max[i] = n5_max[i] + b5[i]
        n5_max[i] = max(0, n5_max[i])
    n5_min = [0] * len(b5)
    for i in range(len(w5)): #for each neuron in layer
        for j in range(len(w5[0])): #for each incoming edge
            if w5[i][j] <= 0:
                n5_min[i] = n5_min[i] + (w5[i][j]*n4_extremum[1][j])
                if ver:
                    print("Layer 5 input for lower bound:", n4_extremum[1][j], "   --node: ", i+1)
            else:
                n5_min[i] = n5_min[i] + (w5[i][j]*n4_extremum[0][j])
                if ver:
                    print("Layer 5 input for lower bound:", n4_extremum[0][j], "   --node: ", i+1)
        n5_min[i] = n5_min[i] + b5[i]
        n5_min[i] = max(0, n5_min[i])
    n5_extremum = np.vstack((n5_min, n5_max))
#    print("Bounds of layer n5\n", n5_extremum, "\n")

    #LAYER6 BOUNDS
    n6_max = [0] * len(b6)
    for i in range(len(w6)): #for each neuron in layer
        for j in range(len(w6[0])): #for each incoming edge
            if w6[i][j] <= 0:
                n6_max[i] = n6_max[i] + (w6[i][j]*n5_extremum[0][j])
                if ver:
                    print("Layer 6 input for upper bound:", n5_extremum[0][j], "   --node: ", i+1)
            else:
                n6_max[i] = n6_max[i] + (w6[i][j]*n5_extremum[1][j])
                if ver:
                    print("Layer 6 input for upper bound:", n5_extremum[1][j], "   --node: ", i+1)
        n6_max[i] = n6_max[i] + b6[i]
        n6_max[i] = max(0, n6_max[i])
    n6_min = [0] * len(b6)
    for i in range(len(w6)): #for each neuron in layer
        for j in range(len(w6[0])): #for each incoming edge
            if w6[i][j] <= 0:
                n6_min[i] = n6_min[i] + (w6[i][j]*n5_extremum[1][j])
                if ver:
                    print("Layer 6 input for lower bound:", n5_extremum[1][j], "   --node: ", i+1)
            else:
                n6_min[i] = n6_min[i] + (w6[i][j]*n5_extremum[0][j])
                if ver:
                    print("Layer 6 input for lower bound:", n5_extremum[0][j], "   --node: ", i+1)
        n6_min[i] = n6_min[i] + b6[i]
        n6_min[i] = max(0, n6_min[i])
    n6_extremum = np.vstack((n6_min, n6_max))
#    print("Bounds of layer n6\n", n6_extremum, "\n")

    #LAYER7 BOUNDS (Output Layer)
    n7_max = [0] * len(b7)
    for i in range(len(w7)): #for each neuron in layer
        for j in range(len(w7[0])): #for each incoming edge
            if w7[i][j] <= 0:
                n7_max[i] = n7_max[i] + (w7[i][j]*n6_extremum[0][j])
                if ver:
                    print("Layer 7 input for upper bound:", n6_extremum[0][j], "   --node: ", i+1)
            else:
                n7_max[i] = n7_max[i] + (w7[i][j]*n6_extremum[1][j])
                if ver:
                    print("Layer 7 input for upper bound:", n6_extremum[1][j], "   --node: ", i+1)
        n7_max[i] = n7_max[i] + b7[i]
        n7_max[i] = (n7_max[i]*out_std_dev_np) + out_mean_np
    n7_min = [0] * len(b7)
    for i in range(len(w7)): #for each neuron in layer
        for j in range(len(w7[0])): #for each incoming edge
            if w7[i][j] <= 0:
                n7_min[i] = n7_min[i] + (w7[i][j]*n6_extremum[1][j])
                if ver:
                    print("Layer 7 input for lower bound:", n6_extremum[1][j], "   --node: ", i+1)
            else:
                n7_min[i] = n7_min[i] + (w7[i][j]*n6_extremum[0][j])
                if ver:
                    print("Layer 7 input for lower bound:", n6_extremum[0][j], "   --node: ", i+1)
        n7_min[i] = n7_min[i] + b7[i]
        n7_min[i] = (n7_min[i]*out_std_dev_np) + out_mean_np
    n7_extremum = np.vstack((n7_min, n7_max))
#    print("Bounds of Output layer (n7)\n", n7_extremum)

    #Storing Bounds
    os.makedirs(os.path.dirname("Datasets/ACAS_Xu/Computed_Output_Bounds/"), exist_ok=True)
    if prop == 1:
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer1_prop1.npy', n1_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer2_prop1.npy', n2_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer3_prop1.npy', n3_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer4_prop1.npy', n4_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer5_prop1.npy', n5_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer6_prop1.npy', n6_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer7_prop1.npy', n7_extremum)
    elif prop == 2:
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer1_prop2.npy', n1_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer2_prop2.npy', n2_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer3_prop2.npy', n3_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer4_prop2.npy', n4_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer5_prop2.npy', n5_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer6_prop2.npy', n6_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer7_prop2.npy', n7_extremum)
    elif prop == 3:
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer1_prop3.npy', n1_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer2_prop3.npy', n2_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer3_prop3.npy', n3_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer4_prop3.npy', n4_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer5_prop3.npy', n5_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer6_prop3.npy', n6_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer7_prop3.npy', n7_extremum)
    elif prop == 4:
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer1_prop4.npy', n1_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer2_prop4.npy', n2_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer3_prop4.npy', n3_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer4_prop4.npy', n4_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer5_prop4.npy', n5_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer6_prop4.npy', n6_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer7_prop4.npy', n7_extremum)
    else:
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer1_complete.npy', n1_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer2_complete.npy', n2_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer3_complete.npy', n3_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer4_complete.npy', n4_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer5_complete.npy', n5_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer6_complete.npy', n6_extremum)
        np.save('Datasets/ACAS_Xu/Computed_Output_Bounds/layer7_complete.npy', n7_extremum)


def remove():
    folderName = "Datasets/ACAS_Xu/Computed_Output_Bounds/"
    try:
        shutil.rmtree(folderName)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))



if __name__ == '__main__':
    print("Import layer_nnet and update number in main.py. \nRun main.py")

