import sys
import os

#import cPickle as pickle
#import _pickle as pickle
import pickle

from cfr.logger import Logger as Log
Log.VERBOSE = True

import cfr.evaluation as evaluation
from cfr.plotting import *

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted

def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = [l.split('=') for l in f.read().split('\n') if '=' in l]
        cfg = dict([(kv[0], eval(kv[1])) for kv in cfg])
    return cfg

def evaluate(config_file, overwrite=False, filters=None):

    if not os.path.isfile(config_file):
        raise Exception('Could not find config file at path: %s' % config_file)

    cfg = load_config(config_file)

    output_dir = cfg['outdir']

    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train = cfg['datadir']+cfg['dataform']
    #print(data_train + " , F")
    data_test = cfg['datadir']+cfg['data_test']
    binary = False
    if cfg['loss'] == 'log':
        binary = True

    # Evaluate results
    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        print(" Doing this")
        eval_results, configs = evaluation.evaluate(output_dir,
                                data_path_train=data_train,
                                data_path_test=data_test,
                                binary=binary)
        # Save evaluation
        print("ITE Prediction Shape: ", np.shape(eval_results['test']['iteff_pred']))
        np.save('results/example_ihdp/res.npy', eval_results)
        pickle.dump(eval_results, open(eval_path, "wb"))
    else:
        if Log.VERBOSE:
            print ('Loading evaluation results from %s...' % eval_path)
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))

    # Experimental plotting for study
    plot_all_profiles(eval_results,patient_nrs=[23, 77, 530, 956, 1200, 1549, 2987, 6903, 7043])


if __name__ == "__main__":

    arg2 = '1'

    config_file = "configs/example_ihdp.txt"

    overwrite = True

    filters = None
    if len(sys.argv)>3:
        filters = eval(sys.argv[3])

    evaluate(config_file, overwrite, filters=filters)


