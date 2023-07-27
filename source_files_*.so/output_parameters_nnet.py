import input_parameters_nnet as ipara


def out_para(nnet):
    mean, std_dev, lb, ub = ipara.parameters(open(nnet, "r"))

    #For Inverse Normalization
    out_mean_np = mean[0][len(mean[0])-1]
    out_std_dev_np = std_dev[0][len(std_dev[0])-1]

    return out_mean_np,out_std_dev_np



if __name__ == '__main__':
    print("Run main.py")
