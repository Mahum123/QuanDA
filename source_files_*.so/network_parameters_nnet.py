import sys
import array


def parameters(nnet):

    #INITIALIZE EMPTY ARRAYS FOR NETWORK PARAMETERS
    w1 = []
    w2 = []
    w3 = []
    w4 = []
    w5 = []
    w6 = []
    w7 = []
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    b5 = []
    b6 = []
    b7 = []

    #READING AND STORING NETWORK PARAMETERS IN VARIABLES
    for i, line in enumerate(nnet):

    #Hidden Layer 1
        if i in range(10,60):
            w1.append(line[:-2])
        if i in range(60,110):
            b1.append(float(line[:-2]))
            
    #Hidden Layer 2
        if i in range(110,160):
            w2.append(line[:-2])
        if i in range(160,210):
            b2.append(float(line[:-2]))
            
    #Hidden Layer 3
        if i in range(210,260):
            w3.append(line[:-2])
        if i in range(260,310):
            b3.append(float(line[:-2]))
            
    #Hidden Layer 4
        if i in range(310,360):
            w4.append(line[:-2])
        if i in range(360,410):
            b4.append(float(line[:-2]))
            
    #Hidden Layer 5
        if i in range(410,460):
            w5.append(line[:-2])
        if i in range(460,510):
            b5.append(float(line[:-2]))
            
    #Hidden Layer 6
        if i in range(510,560):
            w6.append(line[:-2])    
        if i in range(560,610):
            b6.append(float(line[:-2]))
            
    #Output Layer
        if i in range(610,615):
            w7.append(line[:-2]) #removing \n and , at the end of the string
        if i in range(615,620):
            b7.append(float(line[:-2]))
    
    #Changing string weight arrays to float weight arrays
    rows = len(w1) #number of rows in matrix
    columns = len(w1[0].split(",")) #number of columns in matrix
    for row in range(rows):
        w1[row] = w1[row].split(",")
        for column in range(columns):
            w1[row][column] = float(w1[row][column])

    rows = len(w2) #number of rows in matrix
    columns = len(w2[0].split(",")) #number of columns in matrix
    for row in range(rows):
        w2[row] = w2[row].split(",")
        for column in range(columns):
            w2[row][column] = float(w2[row][column])
            
    rows = len(w3) #number of rows in matrix
    columns = len(w3[0].split(",")) #number of columns in matrix
    for row in range(rows):
        w3[row] = w3[row].split(",")
        for column in range(columns):
            w3[row][column] = float(w3[row][column])
        
    rows = len(w4) #number of rows in matrix
    columns = len(w4[0].split(",")) #number of columns in matrix
    for row in range(rows):
        w4[row] = w4[row].split(",")
        for column in range(columns):
            w4[row][column] = float(w4[row][column])
            
    rows = len(w5) #number of rows in matrix
    columns = len(w5[0].split(",")) #number of columns in matrix
    for row in range(rows):
        w5[row] = w5[row].split(",")
        for column in range(columns):
            w5[row][column] = float(w5[row][column])
    
    rows = len(w6) #number of rows in matrix
    columns = len(w6[0].split(",")) #number of columns in matrix
    for row in range(rows):
        w6[row] = w6[row].split(",")
        for column in range(columns):
            w6[row][column] = float(w6[row][column])
            
    rows = len(w7) #number of rows in matrix
    columns = len(w7[0].split(",")) #number of columns in matrix
    for row in range(rows):
        w7[row] = w7[row].split(",")
        for column in range(columns):
            w7[row][column] = float(w7[row][column])
            
#    print(len(w1),len(w1[0]))
#    print(w7[4])
#    print(b7)

    return w1, w2, w3, w4, w5, w6, w7, b1, b2, b3, b4, b5, b6, b7


if __name__ == '__main__':
    print("Run main.py")

