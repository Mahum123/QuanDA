import os
import shutil
import numpy as np


def generate_log(nnet,prop):

    #SOURCE DIRECTORY
    dirSource = "Datasets/ACAS_Xu/"

    #DESTINATION DIRECTIORY
    dirDest = nnet[nnet.find('ACASXU'):]
    dirDest = dirDest[:-5]
    dirDest = "Logs/ACAS_Xu/" + dirDest + "/"
#    if os.path.exists(dirDest):
#        shutil.rmtree(dirDest)
#    os.makedirs(dirDest)

    #DETAILED LOG
    if prop==0:
        for fileName in os.listdir(dirSource+"Probablity_Estimates/"):
            source = dirSource + "Probablity_Estimates/" + fileName
            destination = dirDest + "Detailed_CompleteNetwork/" + fileName
            if not os.path.exists(dirDest+"Detailed_CompleteNetwork/"):
                os.makedirs(dirDest+"Detailed_CompleteNetwork/")
            shutil.copy(source, destination)
    else:
        for fileName in os.listdir(dirSource+"Probablity_Estimates/"):
            source = dirSource + "Probablity_Estimates/" + fileName
            destination = dirDest + "Detailed_Prop/" + fileName 
            if not os.path.exists(dirDest+"Detailed_Prop/"):
                os.makedirs(dirDest+"Detailed_Prop/")
            shutil.copy(source, destination)


    #SUMMARY
    if prop==0:
        shutil.copy(dirSource+"Summary.txt", dirDest+"Summary_CompleteNetwork.txt")
    else:
        shutil.copy(dirSource+"Summary.txt", dirDest+"Summary_Detailed_Prop"+str(prop)+".txt")




def write_summ(nnet,iterations,prop,bounds,fileNameProbab,time,delta,eps,ver):

    fileSumm = open("Datasets/ACAS_Xu/Summary.txt","w")

    #Name
    netName = nnet[nnet.find('ACASXU'):]
    netName = netName[:-5]
    fileSumm.writelines(["Network: ",netName,"\n"])

    #Property
    if prop==0:
        fileSumm.write("Property: None\n")
    else:
        fileSumm.writelines(["Property: ",str(prop),"\n"])

    #Stats
    if iterations==None:
        fileSumm.writelines(["Confidence Level: ",str(delta*100),"%\n"])
        fileSumm.writelines(["Difference between True and Estimated Probabilities: ",str(eps*100),"%\n"])
        fileSumm.writelines(["Execution Time for Experiment: ",str(time),"\n"])
        fileSumm.write("\n======================================================================\n\n")
    else:
        fileSumm.writelines(["Number of iterations: ",str(iterations),"\n"])
        fileSumm.writelines(["Execution Time for Experiment: ",str(time),"\n"])
        fileSumm.write("\n======================================================================\n\n")

    #Probability Estimates
    probab = np.load(fileNameProbab)
    if iterations==1:
        acc_probab = probab #single estimate available
    else:
        acc_probab = np.sum(probab, axis=2)
        acc_probab = acc_probab/probab.shape[2] #output x segments

    fileSumm.write("Output[0]: Clear-of-conflict\n")
    fileSumm.write("Output[1]: Weak Right\n")
    fileSumm.write("Output[2]: Strong Right\n")
    fileSumm.write("Output[3]: Weak Leaft\n")
    fileSumm.write("Output[4]: Strong Left\n")
    fileSumm.write("\n======================================================================\n\n")

    if prop==0:
        fileSumm.write("Probability Estimates: \n\n")
        for node in range(bounds.shape[0]):
            fileSumm.writelines(["Output[",str(node),"]: \n"])
            if ver:
                print("Output[",node,"]: \n")
            for seg in range(bounds.shape[1]):
                if acc_probab[node,seg]!=0:
                    fileSumm.writelines(["     Output range: ",str(bounds[node,seg,:]),"\tProbability: ",str(acc_probab[node,seg]),"\n"])
                    if ver:
                        print("    Output range: ",bounds[node,seg,:],"\tProbability: ",acc_probab[node,seg])
        print("Done.")

    elif prop==1:
        fileSumm.write("Probability Estimates for Property 1 (ACAS Xu): ")
        fileSumm.write("\t Output[0] is at most 1500\n\n")
        fileSumm.writelines(["Estimated Probability for the property to hold: ",str(acc_probab),"\n\n"])
        fileSumm.write("Estimates per iteration...\n")
        for p in range(probab.shape[0]):
            fileSumm.writelines(["\tIteration ",str(p+1),":\t",str(probab[p]),"\n"])
        print("Estimated Probability for Property 1 to hold:", acc_probab)

    elif prop==2:
        fileSumm.write("Probability Estimates for Property 2 (ACAS Xu): ")
        fileSumm.write("\t Output[0] is not maximal\n\n")
        fileSumm.writelines(["Estimated Probability for the property to hold: ",str(acc_probab),"\n\n"])
        fileSumm.write("Estimates per iteration...\n")
        for p in range(probab.shape[0]):
            fileSumm.writelines(["\tIteration ",str(p+1),":\t",str(probab[p]),"\n"])
        print("Estimated Probability for Property 2 to hold:", acc_probab)


    elif prop==3:
        fileSumm.write("Probability Estimates for Property 3 (ACAS Xu): ")
        fileSumm.write("\t Output[0] is not minimal\n\n")
        fileSumm.writelines(["Estimated Probability for the property to hold: ",str(acc_probab),"\n\n"])
        fileSumm.write("Estimates per iteration...\n")
        for p in range(probab.shape[0]):
            fileSumm.writelines(["\tIteration ",str(p+1),":\t",str(probab[p]),"\n"])
        print("Estimated Probability for Property 3 to hold:", acc_probab)

    elif prop==4:
        fileSumm.write("Probability Estimates for Property 4 (ACAS Xu): ")
        fileSumm.write("\t Output[0] is not minimal\n\n")
        fileSumm.writelines(["Estimated Probability for the property to hold: ",str(acc_probab),"\n\n"])
        fileSumm.write("Estimates per iteration...\n")
        for p in range(probab.shape[0]):
            fileSumm.writelines(["\tIteration ",str(p+1),":\t",str(probab[p]),"\n"])
        print("Estimated Probability for Property 4 to hold:", acc_probab)

    elif prop==999:
        fileSumm.write("Probability Estimates for Robustness under Noise (ACAS Xu): ")
        fileSumm.write("\t Output[0] is minimal\n\n")
        fileSumm.writelines(["Estimated Probability for the property to hold: ",str(acc_probab),"\n\n"])
        fileSumm.write("Estimates per iteration...\n")
        for p in range(probab.shape[0]):
            fileSumm.writelines(["\tIteration ",str(p+1),":\t",str(probab[p]),"\n"])
        print("Estimated Probability for Property to hold:", acc_probab)

    fileSumm.close()




def remove():
    shutil.rmtree("Datasets/ACAS_Xu/Computed_Output_Bounds")
    shutil.rmtree("Datasets/ACAS_Xu/Computed_Output_Bounds_Groups")
    shutil.rmtree("Datasets/ACAS_Xu/Probablity_Estimates")
    os.remove("Datasets/ACAS_Xu/Summary.txt")



    
if __name__ == '__main__':
    print("Run main.py")


