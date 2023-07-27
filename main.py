import time
import argparse
import os
import errno
import hoef_size as hsize


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point number" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', type=int, help='Cuda device index')
    parser.add_argument('-d', '--delta', type=restricted_float, help='Confidence interval for the Probability estimate: [0,1)')
    parser.add_argument('-e', '--eps', type=restricted_float, help='Maximum difference between true and estimated Probabilities: (0,1]')
    parser.add_argument('-i', '--iterations', type=int, help='Number of iterations to estimate probabilities. This overrides the number of iterations required to achieve user determined "Confidence"')
    parser.add_argument('-n', '--network', required=True, help='Path for network (.nnet) file')
    parser.add_argument('-p', '--prop', metavar='N', type=int, help='Property index: [1,2,3]')
    parser.add_argument('-s', '--seg', type=int, help='Number of stratum to split the input bounds of each network node')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


def probab_estimate(network,iterations,delta,eps,prop,seg,dev,ver):

    if iterations==None:
        ite = hsize.hoef_size(delta,eps)
        ite = int(ite)
        print("\n-----------------------------------------------------------------------------------------")
        print("   Required number of experiments (iterations) for")
        print("        Confidence level: ", delta, "(", delta*100, "%)")
        print("        Difference in true and estimated probabilities: ", eps, "(", eps*100, "%)")
        print("             == ",ite)
        print("-----------------------------------------------------------------------------------------\n")
        print("Initiating experiments...")
    else:
        ite = iterations
        print("\n-----------------------------------------------------------------------------------------")
        print("   User-defined number of iterations for probability estimate == ",ite)
        print("-----------------------------------------------------------------------------------------\n")
        print("Initiating experiments...")

    #Assigning Number of layers for the desired network
    if "ACAS_Xu" in network:
        layers = 7 	#6 Hidden, 1 Output

    lay.compute_complete_bounds(network,prop,ver) #bounds stored in Directory
    compute_group_bounds_flag = True

    for i in range(ite):
        print("Iteration: ",i+1,"...")
        for l in range(layers):
            number_of_nodes_input, number_of_nodes_output = lay.number_of_nodes(network,l) #ACAS Xu

            #SPLIT NODES OF THE LAYER INTO GROUPS
            if number_of_nodes_input>10: 
                temp = 10
                for t in range(10):
                    if number_of_nodes_input%temp==0:
                        groups_layer = number_of_nodes_input//temp #number of groups
                        break
                    temp = temp-1
                    if temp==0: 
                        raise NotImplementedError("Need to make groups with unequal number of nodes... Not supported with current version of the framework")
            else: # number of nodes in layer <= 10
                groups_layer = 1

            #ESTIMATE PROBABILITIES FOR EACH GROUP OF THE LAYER
            if groups_layer==1:
                probab.estimate_probab(network,prop,l,seg,None,None,number_of_nodes_input,number_of_nodes_output,0,None,dev) #save probab estimate
            else:
                probab.group_bounds(network,prop,l,seg,groups_layer,number_of_nodes_input,compute_group_bounds_flag,dev) #individual group results
                fileName, output_seg = conc_probab.estimate_probab(network,prop,l,seg,number_of_nodes_output,groups_layer,dev) #complete layer results
        compute_group_bounds_flag = False

    return fileName, output_seg




if __name__ == '__main__':
    start_time = time.process_time()
    args = parse_args()
    if "ACAS_Xu" in args.network:
        import layer_nnet as lay #for ACAS Xu
        import probab_estimate_nnet as probab #for ACAS Xu
        import probab_estimate_group_input_nnet as conc_probab #for ACAS Xu
        import cleanup_nnet as fclean #for ACAS Xu

    #SET DEFAULT CONFIDENCE VALUES AND EXCEPTIONAL HANDLING
    if args.eps==None:
        eps = 0.05
    else:
        if args.eps==0:
            raise ValueError("-e/--eps can not be 0.0") #eps=0 requires infinite iterations
        else:
            eps = args.eps
    if args.delta==None:
        delta = 0.95
    else:
        if args.delta==1:
            raise ValueError("-d/--delta can not be 1.0") #delta=1 implies 100% confidence
        else:
            delta = args.delta
    if not os.path.isfile(args.network):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.network)

    #SET DEFAULT NUMBER OF SEGMENTS TO SPLIT THE BOUNDS OF EACH NETWORK NODE
    if args.seg==None:
        seg = 5
    elif args.seg<1:
            raise ValueError("-s/--seg can not be less than 1") #minimum seg=1, which implies no splitting of the bounds
    else:
        seg = args.seg


    #IDENTIFY PROPERTY TO CHECK (for desired neural network)
    if "ACAS_Xu" in args.network:
        if args.prop==None:
            prop = 0
        elif args.prop < 1 or args.prop > 3:
            raise ValueError("-p/--prop can not be greater than 3")
        elif args.prop==3:
            prop = 4
        else:
            prop = args.prop


    #RUN
    fileName, output_seg = probab_estimate(args.network,args.iterations,delta,eps,prop,seg,args.cuda,args.verbose)
    exec_time = time.process_time()-start_time
    fclean.write_summ(args.network,args.iterations,prop,output_seg,fileName,exec_time,delta,eps,args.verbose)
    fclean.generate_log(args.network,prop)
    fclean.remove()

    print("\nExecution Time: ",exec_time)
    if prop==4:
        print("\nTo see detailed results, check: .\\Logs \n(Note that Prop4 in file names is a typo...the results correspond to property 3 from the paper)")
    else:
        print("\nTo see detailed results, check: .\\Logs")







