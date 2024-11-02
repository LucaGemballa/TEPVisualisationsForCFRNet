import tensorflow as tf
import numpy as np
import sys, os
import getopt
import random
import datetime
import traceback

import cfr.cfr_net as cfr
from cfr.util import *

''' Define parameter flags '''
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.flags.DEFINE_integer('n_in', 2, """Number of representation layers. """)
tf.flags.DEFINE_integer('n_out', 2, """Number of regression layers. """)
tf.flags.DEFINE_float('p_alpha', 1e-4, """Imbalance regularization param. """)
tf.flags.DEFINE_float('p_lambda', 0.0, """Weight decay regularization parameter. """)
tf.flags.DEFINE_integer('rep_weight_decay', 1, """Whether to penalize representation layers with weight decay""")
tf.flags.DEFINE_float('dropout_in', 0.9, """Input layers dropout keep rate. """)
tf.flags.DEFINE_float('dropout_out', 0.9, """Output layers dropout keep rate. """)
tf.flags.DEFINE_string('nonlin', 'relu', """Kind of non-linearity. Default relu. """)
tf.flags.DEFINE_float('lrate', 0.05, """Learning rate. """)
tf.flags.DEFINE_float('decay', 0.5, """RMSProp decay. """)
tf.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.flags.DEFINE_integer('dim_in', 100, """Pre-representation layer dimensions. """)
tf.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.flags.DEFINE_string('normalization', 'none', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.flags.DEFINE_integer('iterations', 2000, """Number of iterations. """)
tf.flags.DEFINE_float('weight_init', 0.01, """Weight initialization scale. """)
tf.flags.DEFINE_float('lrate_decay', 0.95, """Decay of learning rate every 100 iterations """)
tf.flags.DEFINE_integer('wass_iterations', 20, """Number of iterations in Wasserstein computation. """)
tf.flags.DEFINE_float('wass_lambda', 1, """Wasserstein lambda. """)
tf.flags.DEFINE_integer('wass_bpt', 0, """Backprop through T matrix? """)
tf.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.flags.DEFINE_string('outdir', '../results/tfnet_topic/alpha_sweep_22_d100/', """Output directory. """)
tf.flags.DEFINE_string('datadir', '../data/topic/csv/', """Data directory. """)
tf.flags.DEFINE_string('dataform', 'topic_dmean_seed_%d.csv', """Training data filename form. """)
tf.flags.DEFINE_string('data_test', '', """Test data filename form. """)
tf.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.flags.DEFINE_integer('use_p_correction', 1, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.flags.DEFINE_string('optimizer', 'RMSProp', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.flags.DEFINE_string('imb_fun', 'mmd_lin', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.flags.DEFINE_integer('output_csv',0,"""Whether to save a CSV file with the results""")
tf.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)
tf.flags.DEFINE_integer('pred_output_delay', -1, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.flags.DEFINE_integer('debug', 0, """Debug mode. """)
tf.flags.DEFINE_integer('save_rep', 0, """Save representations after training. """)
tf.flags.DEFINE_float('val_part', 0, """Validation part. """)
tf.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers between treated and control. """)
tf.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)

if FLAGS.sparse:
    import scipy.sparse as sparse

NUM_ITERATIONS_PER_DECAY = 100

__DEBUG__ = False
if FLAGS.debug:
    __DEBUG__ = True

def train(CFR, sess, train_step, D, I_valid, D_test, logfile, i_exp):
    """ Trains a CFR model on supplied data """

    ''' Train/validation split '''
    n = D['x'].shape[0]
    I = range(n); I_train = list(set(I)-set(I_valid))
    n_train = len(I_train)

    ''' Compute treatment probability'''
    p_treated = np.mean(D['t'][I_train])

    #formerly used the len element of reshape
    feedt = D['t'][I_train].reshape(len(D['t'][I_train]),1)
    feedy = D['yf'][I_train].reshape(len(D['yf'][I_train]),1)

    feedtvalid = D['t'][I_valid].reshape(len(D['t'][I_valid]),1)
    feedyvalid = D['yf'][I_valid].reshape(len(D['yf'][I_valid]),1)

    ''' Set up loss feed_dicts'''
    dict_factual = {CFR.x: D['x'][I_train], CFR.t: feedt, CFR.y_: feedy, \
      CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
      CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    if FLAGS.val_part > 0:
        dict_valid = {CFR.x: D['x'][I_valid], CFR.t: feedtvalid, CFR.y_: feedyvalid, \
          CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, \
          CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}

    if D['HAVE_TRUTH']:
        dict_cfactual = {CFR.x: D['x'][I_train], CFR.t: 1-D['t'][I_train], CFR.y_: D['ycf'][I_train], \
          CFR.do_in: 1.0, CFR.do_out: 1.0}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Set up for storing predictions '''
    preds_train = []
    preds_test = []

    ''' Compute losses '''
    losses = []
    obj_loss, f_error, imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],\
      feed_dict=dict_factual)

    cf_error = np.nan
    if D['HAVE_TRUTH']:
        cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

    valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
    if FLAGS.val_part > 0:
        valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],\
          feed_dict=dict_valid)

    losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])

    objnan = False

    reps = []
    reps_test = []

    ''' Train for multiple iterations '''
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        I = random.sample(range(0, n_train), FLAGS.batch_size)
        x_batch = D['x'][I_train,:][I,:]
        t_batch = D['t'][I_train][I]
        y_batch = D['yf'][I_train][I]

        t_batch = t_batch.reshape(len(t_batch),1)
        y_batch = y_batch.reshape(len(y_batch),1)

        if __DEBUG__:
            M = sess.run(cfr.pop_dist(CFR.x, CFR.t), feed_dict={CFR.x: x_batch, CFR.t: t_batch})
            log(logfile, 'Median: %.4g, Mean: %.4f, Max: %.4f' % (np.median(M.tolist()), np.mean(M.tolist()), np.amax(M.tolist())))

        ''' Do one step of gradient descent '''
        if not objnan:
            sess.run(train_step, feed_dict={CFR.x: x_batch, CFR.t: t_batch, \
                CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated})

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(CFR.weights_in[0]), 1)
            sess.run(CFR.projection, feed_dict={CFR.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],
                feed_dict=dict_factual)

            rep = sess.run(CFR.h_rep_norm, feed_dict={CFR.x: D['x'], CFR.do_in: 1.0})
            rep_norm = np.mean(np.sqrt(np.sum(np.square(rep), 1)))

            cf_error = np.nan
            if D['HAVE_TRUTH']:
                cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

            valid_obj = np.nan; valid_imb = np.nan; valid_f_error = np.nan;
            if FLAGS.val_part > 0:
                valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_valid)

            losses.append([obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj])
            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f' \
                        % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)

            if FLAGS.loss == 'log':
                y_pred = sess.run(CFR.output, feed_dict={CFR.x: x_batch, \
                    CFR.t: t_batch, CFR.do_in: 1.0, CFR.do_out: 1.0})
                y_pred = 1.0*(y_pred > 0.5)
                acc = 100*(1 - np.mean(np.abs(y_batch - y_pred)))
                loss_str += ',\tAcc: %.2f%%' % acc

            log(logfile, loss_str)

            if np.isnan(obj_loss):
                log(logfile,'Experiment %d: Objective is NaN. Skipping.' % i_exp)
                objnan = True

        ''' Compute predictions every M iterations '''
        if (FLAGS.pred_output_delay > 0 and i % FLAGS.pred_output_delay == 0) or i==FLAGS.iterations-1:

            Dttrain = D['t'].reshape(len(D['t']),1)
            Dttest = D_test['t'].reshape(len(D_test['t']),1)

            y_pred_f = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                CFR.t: Dttrain, CFR.do_in: 1.0, CFR.do_out: 1.0})
            y_pred_cf = sess.run(CFR.output, feed_dict={CFR.x: D['x'], \
                CFR.t: 1-Dttrain, CFR.do_in: 1.0, CFR.do_out: 1.0})
            preds_train.append(np.concatenate((y_pred_f, y_pred_cf),axis=1))

            if D_test is not None:
                y_pred_f_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                    CFR.t: Dttest, CFR.do_in: 1.0, CFR.do_out: 1.0})
                y_pred_cf_test = sess.run(CFR.output, feed_dict={CFR.x: D_test['x'], \
                    CFR.t: 1-Dttest, CFR.do_in: 1.0, CFR.do_out: 1.0})
                preds_test.append(np.concatenate((y_pred_f_test, y_pred_cf_test),axis=1))

            if FLAGS.save_rep and i_exp == 1:
                reps_i = sess.run([CFR.h_rep], feed_dict={CFR.x: D['x'], \
                    CFR.do_in: 1.0, CFR.do_out: 0.0})
                reps.append(reps_i)

                if D_test is not None:
                    reps_test_i = sess.run([CFR.h_rep], feed_dict={CFR.x: D_test['x'], \
                        CFR.do_in: 1.0, CFR.do_out: 0.0})
                    reps_test.append(reps_test_i)

    return losses, preds_train, preds_test, reps, reps_test

def run(outdir):
    """ Runs an experiment and stores result in outdir """

    ''' Set up paths and start log '''
    npzfile = outdir+'result'
    npzfile_test = outdir+'result.test'
    repfile = outdir+'reps'
    repfile_test = outdir+'reps.test'
    outform = outdir+'y_pred'
    outform_test = outdir+'y_pred.test'
    lossform = outdir+'loss'
    logfile = outdir+'log.txt'
    f = open(logfile,'w')
    f.close()
    dataform = FLAGS.datadir + FLAGS.dataform

    print("Running Train")

    has_test = False
    if not FLAGS.data_test == '': # if test set supplied
        has_test = True
        dataform_test = FLAGS.datadir + FLAGS.data_test

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed) 
    #tf.random.set_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Save parameters '''
    save_config(outdir+'config.txt')

    log(logfile, 'Training with hyperparameters: alpha=%.2g, lambda=%.2g' % (FLAGS.p_alpha,FLAGS.p_lambda))

    ''' Load Data '''
    npz_input = False
    if dataform[-3:] == 'npz':
        npz_input = True
    if npz_input:
        datapath = dataform
        if has_test:
            datapath_test = dataform_test
    else:
        datapath = dataform % 1
        if has_test:
            datapath_test = dataform_test % 1

    log(logfile,     'Training data: ' + datapath)
    if has_test:
        log(logfile, 'Test data:     ' + datapath_test)
    print("should load data now")
    a = np.load(datapath, allow_pickle=True)
    for k in a:
        print(k)

    print(a['arr_0'].shape)
    #print(a['t'].shape)
    D = load_data(datapath)
    D_test = None
    if has_test:
        D_test = load_data(datapath_test)

    log(logfile, 'Loaded data with shape [%d,%d]' % (D['n'], D['dim']))

    ''' Start Session '''
    sess = tf.Session()

    ''' Initialize input placeholders '''
    x  = tf.placeholder("float", shape=[None, D['dim']], name='x') # Features
    t  = tf.placeholder("float", shape=[None, 1], name='t')   # Treatent
    y_ = tf.placeholder("float", shape=[None, 1], name='y_')  # Outcome

    ''' Parameter placeholders '''
    r_alpha = tf.placeholder("float", name='r_alpha')
    r_lambda = tf.placeholder("float", name='r_lambda')
    do_in = tf.placeholder("float", name='dropout_in')
    do_out = tf.placeholder("float", name='dropout_out')
    p = tf.placeholder("float", name='p_treated')

    ''' Define model graph '''
    log(logfile, 'Defining graph...\n')
    dims = [D['dim'], FLAGS.dim_in, FLAGS.dim_out]
    CFR = cfr.cfr_net(x, t, y_, p, FLAGS, r_alpha, r_lambda, do_in, do_out, dims)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, \
        NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)

    opt = None
    if FLAGS.optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(lr)
    elif FLAGS.optimizer == 'GradientDescent':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(lr)
    else:
        opt = tf.train.RMSPropOptimizer(lr, FLAGS.decay)

    ''' Unused gradient clipping '''
    #gvs = opt.compute_gradients(CFR.tot_loss)
    #capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
    #train_step = opt.apply_gradients(capped_gvs, global_step=global_step)

    train_step = opt.minimize(CFR.tot_loss,global_step=global_step)

    ''' Set up for saving variables '''
    all_losses = []
    all_preds_train = []
    all_preds_test = []
    all_valid = []
    if FLAGS.varsel:
        all_weights = None
        all_beta = None

    all_preds_test = []

    ''' Handle repetitions '''
    n_experiments = FLAGS.experiments
    if FLAGS.repetitions>1:
        if FLAGS.experiments>1:
            log(logfile, 'ERROR: Use of both repetitions and multiple experiments is currently not supported.')
            sys.exit(1)
        n_experiments = FLAGS.repetitions

    ''' Run for all repeated experiments '''
    for i_exp in range(1,n_experiments+1):

        if FLAGS.repetitions>1:
            log(logfile, 'Training on repeated initialization %d/%d...' % (i_exp, FLAGS.repetitions))
        else:
            log(logfile, 'Training on experiment %d/%d...' % (i_exp, n_experiments))

        ''' Load Data (if multiple repetitions, reuse first set)'''

        if i_exp==1 or FLAGS.experiments>1:
            D_exp_test = None
            if npz_input:
                D_exp = {}
                D_exp['x']  = D['x'][:,:]
                D_exp['t']  = D['t']
                D_exp['yf'] = D['yf']
                if D['HAVE_TRUTH']:
                    D_exp['ycf'] = D['ycf']
                else:
                    D_exp['ycf'] = None

                if has_test:
                    D_exp_test = {}
                    D_exp_test['x']  = D_test['x']
                    D_exp_test['t']  = D_test['t']
                    D_exp_test['yf'] = D_test['yf']
                    if D_test['HAVE_TRUTH']:
                        D_exp_test['ycf'] = D_test['ycf']
                    else:
                        D_exp_test['ycf'] = None
            else:
                datapath = dataform % i_exp
                D_exp = load_data(datapath)
                if has_test:
                    datapath_test = dataform_test % i_exp
                    D_exp_test = load_data(datapath_test)

            D_exp['HAVE_TRUTH'] = D['HAVE_TRUTH']
            if has_test:
                D_exp_test['HAVE_TRUTH'] = D_test['HAVE_TRUTH']

        ''' Split into training and validation sets '''
        I_train, I_valid = validation_split(D_exp, FLAGS.val_part)

        ''' Run training loop '''
        losses, preds_train, preds_test, reps, reps_test = \
            train(CFR, sess, train_step, D_exp, I_valid, \
                D_exp_test, logfile, i_exp)


        ''' Collect all reps '''
        all_preds_train.append(preds_train)
        all_preds_test.append(preds_test)
        all_losses.append(losses)

        ''' Fix shape for output (n_units, dim, n_reps, n_outputs) '''
        out_preds_train = np.swapaxes(np.swapaxes(all_preds_train,1,3),0,2)
        if  has_test:
            out_preds_test = np.swapaxes(np.swapaxes(all_preds_test,1,3),0,2)
        out_losses = np.swapaxes(np.swapaxes(all_losses,0,2),0,1)

        ''' Store predictions '''
        log(logfile, 'Saving result to %s...\n' % outdir)
        if FLAGS.output_csv:
            np.savetxt('%s_%d.csv' % (outform,i_exp), preds_train[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (outform_test,i_exp), preds_test[-1], delimiter=',')
            np.savetxt('%s_%d.csv' % (lossform,i_exp), losses, delimiter=',')

        ''' Compute weights if doing variable selection '''
        if FLAGS.varsel:
            if i_exp == 1:
                all_weights = sess.run(CFR.weights_in[0])
                all_beta = sess.run(CFR.weights_pred)
            else:
                all_weights = np.dstack((all_weights, sess.run(CFR.weights_in[0])))
                all_beta = np.dstack((all_beta, sess.run(CFR.weights_pred)))

        ''' Save results and predictions '''
        all_valid.append(I_valid)
        if FLAGS.varsel:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, w=all_weights, beta=all_beta, val=np.array(all_valid))
        else:
            np.savez(npzfile, pred=out_preds_train, loss=out_losses, val=np.array(all_valid))

        if has_test:
            np.savez(npzfile_test, pred=out_preds_test)

        ''' Save representations '''
        if FLAGS.save_rep and i_exp == 1:
            np.savez(repfile, rep=reps)

            if has_test:
                np.savez(repfile_test, rep=reps_test)

    # intermediate step for shap testing
    """print(CFR.weights_in[0].shape)
    input_as_tensor = tf.convert_to_tensor(CFR.weights_in[0], dtype=tf.float32)
    output_as_tensor = tf.convert_to_tensor(CFR.weights_pred, dtype=tf.float32)
    print(input_as_tensor,input_as_tensor.shape)
    print(output_as_tensor,output_as_tensor.shape)

    shap_input = np.transpose(D_exp['x'], axes = [1,0])

    background = D_exp['x'][np.random.choice(D_exp['x'].shape[0], 200, replace=False)]
    background = np.transpose(background, axes = [1,0])
    #background = D_exp['x'][np.random.choice(D_exp['x'].shape[0], 100, replace=False)]

    outshape_cut = output_as_tensor[-1]
    print("cut ", outshape_cut.shape )
    print(background.shape)

    print(shap_input.shape)

    print(D_exp.keys())
    print(type(D_exp['x']))
    print(D_exp['t'].shape)
    print(D_exp['yf'].shape)
    print(D_exp['ycf'].shape)
    #print(tf.shape(D_exp))
    print("outshape ", output_as_tensor.shape)
    print("inshape ", input_as_tensor.shape)
    #explainer = shap.DeepExplainer((shap_input.shape,D_exp['t']),shap_input) # CFR is an instance of cfr_net
    explainer = shap.DeepExplainer((input_as_tensor,output_as_tensor),background)

    expl_input = D_exp['x'][0:200]
    expl_input = np.transpose(expl_input, axes = [1,0])
    print("expl " ,expl_input.shape)
    shap_values = explainer.shap_values(expl_input)
    """
    #shap.plots.waterfall(shap_values[0])

def main(argv=None):  # pylint: disable=unused-argument
    """ Main entry point """
    print("Main Func Train")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
    outdir = FLAGS.outdir+'/results_'+timestamp+'/'
    os.mkdir(outdir)

    try:
        run(outdir)
    except Exception as e:
        with open(outdir+'error.txt','w') as errfile:
            errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
        raise

if __name__ == '__main__':
    print("Entry")
    tf.app.run()
