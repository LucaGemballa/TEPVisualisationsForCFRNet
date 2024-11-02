# TEPVisualisationsForCFRNet

This repository uses a minimaly modified version of the cfr_net code provided through: https://github.com/clinicalml/cfrnet
This readme file up until Information for this specific use case is copied from their work so you don't have to look it up seperately.

# cfrnet
Counterfactual regression (CFR) by learning balanced representations, as developed by Johansson, Shalit & Sontag (2016) and Shalit, Johansson & Sontag (2016). cfrnet is implemented in Python using TensorFlow 0.12.0-rc1 and NumPy 1.11.3. The code has _not_ been tested with TensorFlow 1.0.

# Code

The core components of cfrnet, i.e. the TensorFlow graph, is contained in cfr/cfr_net.py. The training is performed by cfr_net_train.py. The file cfr_param_search.py takes a configuration file as input and allows the user to randomly sample from the supplied parameters (given that there are multiple values given in a list. See configs/example_ihdp.txt for an example.

A typical experiment uses cfr_param_search.py and evaluate.py as sub-routines. cfr_param_search is best used to randomly search the parameter space. In the output directory, it creates a log of which configurations have been used so far, so that the same experiment is not repeated. evaluate.py goes through the predictions produced by the model and evaluates the error.

## cfr_param_search

The script _cfr_param_search.py_ runs a random hyperparameter search given a configuration file.

Usage:

```
python cfr_param_search.py <config_file> <num_runs>
```
The _config_file_ argument should contain the path to a text file where each line is a key-value pair for a CFR parameter.

The _num_run_ argument should contain an integer to indicate how many parameter settings should be sampled. If all possible configurations should be used, this can be set arbitrarily high as the script will terminate when all have been used. If the number of possible settings is vast, a smaller value for _num_runs_ may be appropriate.

Example:

```
python evaluate.py configs/example_ihdp.txt 10
```

Example configuration file (from configs/example_ihdp.txt):

```
p_alpha=[0, 1e-1]
p_lambda=[1e-3]
n_in=[2]
n_out=[2]
dropout_in=1.0
...
```

Note that some of the lines have square brackets to indicate lists. If a parameter list contains more than a single element, cfr_param_search will sample uniformly from these values. In this way, random parameter search can be performed.

## evaluate

The script _evaluate.py_ performs an evaluation of a trained model based on the predictions made for the training and test sets.

Usage:

```
python evaluate.py <config_file> [overwrite] [filters]
```

The parameter _config_file_ should be the same as the one used in cfr_param_search. (Note: evaluate only uses the settings for dataform, data_test, datadir and outdir, the rest can be changed without affecting the evaluation.)

If the _overwrite_ parameter is set to "1", the script re-computes all error estimates. If it is set to "0" it re-uses stored values, but re-prints and re-plots all results.

The argument _filters_ accepts a string in the form of a python dict containing values of the parameters the used wishes to filter. This produces plots and text summaries only of results corresponding to configuration that matches the filter.

Example:

```
python evaluate.py configs/example_ihdp.txt 0 "{p_alpha: 0}"
```

# Examples

A simple experiment example is contained in example_ihdp.sh. This file runs the model on (a subset of) the IHDP data with parameters supplied by configs/example_ihdp.txt. The data for this example can be downloaded from http://www.fredjo.com/files/ihdp_npci_1-100.train.npz (train) and http://www.fredjo.com/files/ihdp_npci_1-100.test.npz (test). For the full data (of 1000 replications) used in the ICML 2017 paper, please visit https://www.fredjo.com/.

# FAQ

* Q: **What are the hyperparameters used on IHDP in the ICML 2017 paper?** A: The parameters were those given in example_ihdp.txt but with p_alpha = 0.3
* Q: **I don't get the same IHDP results as in the paper when I try to replicate with the IHDP example from Github.** A: The ICML 2017 results were computed over the full set of 1000 replications. The Github IHDP example uses only 100 examples as it is meant to serve as a quick demo. Please find the 1000 replications at https://www.fredjo.com/. 

# References
Uri Shalit, Fredrik D. Johansson & David Sontag. [Estimating individual treatment effect: generalization bounds and algorithms](https://arxiv.org/abs/1606.03976), 34th International Conference on Machine Learning (ICML), August 2017.

Fredrik D. Johansson, Uri Shalit &  David Sontag. [Learning Representations for Counterfactual Inference](http://jmlr.org/proceedings/papers/v48/johansson16.pdf). 33rd International Conference on Machine Learning (ICML), June 2016.

# Information for this specific use case
If conda does not work at first go to Anaconda Powershell Prompt and use: conda init powershell
Then after re-opening the terminal via conda create -n eapp_env python=3.6.2
and conda activate eapp_env
after this install the packages via pip

For this repository two environments are needed. To use the cfr_net to make predictions the following packages are needed:
    Python 3.6.2
    Numpy 1.19.4
    Matplotlib 3.3.4
    Scipy 1.5.4
    Tensorflow 1.4.0
    tqdm 4.64.1
    pyqt5
    pandas
    tkinter?

## Dataset from SEER
In the original repository IHDP was used for demonstration. For my thesis I constructed a new data set from the SEER Database.

Use data_refinement to modify the unrefined_study_data. You will receive masterstudytest.npz and masterstudytrain.npz to continue with training
These two npz files have to be moved to the data folder

## Training CFRNet and modifications
cfr_param_search does the search specified like above and then calls cfr_net_train

## Evaluation and plotting
For this step we can put aside the evaluation tools coming from the original repository as now our focus is on visualisation
New setup is 
conda create -n eapp_env python=3.8.19
conda activate eapp_env
conda install --force-reinstall pip should help if installing the other packages doesn't work

then:
    Matplotlib 3.7.5
    Numpy 1.24.4
    Pandas 1.4.2
    pyqt5
    scikit-learn 1.3.2
    zepid

In addition to this because the cfrnet is not structured like the usual ML model some tweaks need to be made to the two way partial dependence calculation
At the import from sklearn.inspection import PartialDependenceDisplay
Select PartialDependenceDisplay and click 'Go to type definition'
Replace the content of that file with the content of pdp_changed.txt

Then through evaluate you can call the code for plotting 18 different treatment effect prediction visualisations. In addition for 6 of these visualisations 
you can plot them for any number of patients from the test data


