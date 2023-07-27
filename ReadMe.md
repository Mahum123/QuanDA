# QuanDA: Quantitative Deep Neural Network Analysis

This repository contains the execution code for the quantitative deep neural network (DNN) framework, **QuanDA**, as described in the paper[^*]. The ACAS Xu networks benchmark and its properties are used to show the analysis of DNNs using QuandA.


## This Repository

This directory contains the following:
* `./Datasets/ACAS_Xu/nnet/` : The [open-source](https://github.com/guykatzz/ReluplexCav2017/tree/master/nnet) network files (for ACAS Xu)
* `./Datasets/ACAS_Xu/Input_Bounds/` : The [input bounds](https://arxiv.org/pdf/1702.01135.pdf) corresponding to entire (valid input domain) and its safety properties 
* `./source_files_*.so/` : The original source files corresponding to `./*.so` files 
* `./requirements.txt` : The list of required packages (and their specific versions)
* `./*.py` and `./*.so` : QuanDA implementation modules
* `./main.py` : Python wrapper for execution of QuanDA

## Getting Started

### System Requirements
This is the Ubuntu implementation of QuanDA, requiring both CPU and GPU system capabilities. The code has been tested on:
* Ubuntu 18.04 LTS
* CUDA release 11.6
* Python 3.6

### System Setup
Update  pip to the latest supported version on your system: `pip install --upgrade pip`
Install the required packages: `pip install -r requirements.txt`

## Execution
As described in paper, QuanDA checks for three safety properties, i.e., 1-3. For instance, to run the property 1 for network 1_1:   
*Run with confidence interval and deviation of 0.95 and 0.05, respectively:*
`python main.py -n Datasets/ACAS_Xu/nnet/ACASXU_run2a_1_1_batch_2000.nnet -e 0.95 -d 0.05`
*Run a single iteration:* 
`python main.py -n Datasets/ACAS_Xu/nnet/ACASXU_run2a_1_1_batch_2000.nnet -i 1 -p 1`

### Additional Options
* -c : [*default: 0*] CUDA device index (relevant in case when multiple GPUs are available)
* -s : [*default: 5*] Number of stratum for each node bound 

*For additional help, run:*  `python main.py -h`

## Interpreting Results
Analyzing a network generates `./Logs/ACAS_Xu/ACASXU_run2a_X_X_batch_2000/` directory (with ...X_X... corresponding to the network). This directory contains following results:
* `Detailed_Prop/` : contains .npy files containing probabilities of each stratum, of each node, at each layer of the network
* `Summary_Detailed_*.txt` : Summary of stats and output-layer-results for all iterations

[^*]: Paper documenting details of the work soon to be available online.
