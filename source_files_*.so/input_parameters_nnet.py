import sys
import array


def parameters(nnet):
    #INITIALIZE EMPTY VARIABLES
    lb = []
    ub = []
    mean = []
    std_dev = []

    #READING AND STORING NETWORK PARAMETERS IN VARIABLES
    for i, line in enumerate(nnet):

        if i==6:
            lb.append(line[:-2])
        if i==7:
            ub.append(line[:-2])
        if i==8:
            mean.append(line[:-2])
        if i==9:
            std_dev.append(line[:-2])
    
    #Changing string to float arrays
    rows = len(lb) #number of rows in matrix
    columns = len(lb[0].split(",")) #number of columns in matrix
    for row in range(rows):
        lb[row] = lb[row].split(",")
        for column in range(columns):
            lb[row][column] = float(lb[row][column])

    rows = len(ub) #number of rows in matrix
    columns = len(ub[0].split(",")) #number of columns in matrix
    for row in range(rows):
        ub[row] = ub[row].split(",")
        for column in range(columns):
            ub[row][column] = float(ub[row][column])
            
    rows = len(mean) #number of rows in matrix
    columns = len(mean[0].split(",")) #number of columns in matrix
    for row in range(rows):
        mean[row] = mean[row].split(",")
        for column in range(columns):
            mean[row][column] = float(mean[row][column])
        
    rows = len(std_dev) #number of rows in matrix
    columns = len(std_dev[0].split(",")) #number of columns in matrix
    for row in range(rows):
        std_dev[row] = std_dev[row].split(",")
        for column in range(columns):
            std_dev[row][column] = float(std_dev[row][column])
            
#    print(len(mean),len(mean[0]))
#    print(std_dev[0][5])
#    print(ub)

    return mean, std_dev, lb, ub
