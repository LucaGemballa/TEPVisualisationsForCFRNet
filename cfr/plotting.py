import sys
import os
import numpy as np
import collections
import statistics
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random
import math

from scipy.signal import find_peaks

import matplotlib.image as mpimg
import zepid
from zepid.graphics import EffectMeasurePlot

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay



from cfr.dummy_est import DummyRegressor

from cfr.loader import *

LINE_WIDTH = 2
FONTSIZE_LGND = 8
FONTSIZE = 16

EARLY_STOP_SET_CONT = 'valid'
EARLY_STOP_CRITERION_CONT = 'objective'
CONFIG_CHOICE_SET_CONT = 'valid'
CONFIG_CRITERION_CONT = 'pehe_nn'
CORR_CRITERION_CONT = 'pehe'
CORR_CHOICE_SET_CONT = 'test'

EARLY_STOP_SET_BIN = 'valid'
EARLY_STOP_CRITERION_BIN = 'policy_risk'
CONFIG_CHOICE_SET_BIN = 'valid'
CONFIG_CRITERION_BIN = 'policy_risk'
CORR_CRITERION_BIN = 'policy_risk'
CORR_CHOICE_SET_BIN = 'test'

CURVE_TOP_K = 7



"""
Receives list of all features, the feature to split, 
patient_nr: optional, patient for whom individual values are calculated

Output: mean outcomes for treatment and control predictions, sorted by the respective feature characteristics
"""
def get_feature_split(features, feat_results, to_list, co_list, feature_nr=0, patient_nr= 0, split_ind=0, split=False):
    # need to sort the features into buckets

    feat_x = feat_results[0,0,0,:][:,feature_nr]
    if feature_nr == 0:
        feat_name = 'Age '
        feat_names = ['70-74', '75-79', '80-84', '85+']
    else:  
        feat_name = 'Tumor Grade '
        feat_names = np.unique(feat_x)

    all_featvals = np.unique(feat_x)
    
    x_ticks_list = [['', str(featval), ''] for featval in feat_names]
    nr_buckets = len(all_featvals)

    # need to get treatment and control list first, have to extract them from ycf_p and yf_p
    treatment_outcomes_for_featvals = [[to_list[patient] for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]
    control_outcomes_for_featvals = [[co_list[patient] for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]

    treatment_means = [np.mean(to_list) for to_list in treatment_outcomes_for_featvals]
    control_means = [np.mean(co_list) for co_list in control_outcomes_for_featvals]
    ind_values = [to_list[patient_nr], co_list[patient_nr]]

    return treatment_means, control_means, ind_values, x_ticks_list

def listOfTuples2(l1, l2):
    return list(map(lambda x, y:(x,y), l1, l2))

def listOfTuples3(l1, l2, l3):
    return list(map(lambda x, y, z:(x,y,z), l1, l2, l3))

def listOfTuples4(l1, l2, l3, l4):
        return list(map(lambda x, y, z, a:(x,y,z,a), l1, l2, l3, l4))

def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


"""
Give this method any number of patients and two feature numbers. 
It calculates the data needed for every profile once and then calls sub-methods 
to generate the respective graphs either once or for every patient
"""
def plot_all_profiles(results, patient_nrs=[0], feature1=0, feature2=1):
    
    # these names are self-explanatory, they are calculated for the last 8 training episodes
    avg_ite_per_person = np.mean(results['test']['iteff_pred'][:,0,8:15:,:], axis=(1))
    std_ite_per_person = np.std(results['test']['iteff_pred'][:,0,8:15:,:], axis=(1))


    nr_patients = len(avg_ite_per_person[0])

    # contains treatments (0 or 1) for all patients. needed because predictions are safed as factual or counterfactual!
    treatments_for_visual = results['test']['treatment'][:,0,0,:]

    # averaged over all runs for each individual patient
    treatment_outcome_list = [np.mean(results['test']['yf_p'][:,0,8:15,patient], axis=(1))[0] if treatments_for_visual[0][patient] == 1
    else np.mean(results['test']['ycf_p'][:,0,8:15,patient], axis=(1))[0] for patient in range(len(treatments_for_visual[0]))]
    control_outcome_list = [np.mean(results['test']['yf_p'][:,0,8:15,patient], axis=(1))[0] if treatments_for_visual[0][patient] == 0
    else np.mean(results['test']['ycf_p'][:,0,8:15,patient], axis=(1))[0] for patient in range(len(treatments_for_visual[0]))]

    # create list that specifies which graphs are for individual patients and which are for the entire population? Np!

    # generate data for the feature values
    # Feature 1 is the one used if only one feature is part of the graph / Patient Age

    # feat1_vals comprises the ages/age groups of the training cohort
    feat1_vals = results['test']['features'][0,0,0,:][:,feature1]
    uniq_feat1_vals_nolab = np.unique(feat1_vals) 
    uniq_feat1_vals = np.unique(feat1_vals)  # ['70-74', '75-79', '80-84', '85+']
    feat1val_labs = [str(featval) for featval in uniq_feat1_vals_nolab]

    # prediction outcomes are sorted with outcomes that match in regards to patient feature 1
    treatment_outcomes_for_feat1vals = [[treatment_outcome_list[patient] for patient in np.where(feat1_vals == featval)[0]] for featval in uniq_feat1_vals_nolab]
    control_outcomes_for_feat1vals = [[control_outcome_list[patient] for patient in np.where(feat1_vals == featval)[0]] for featval in uniq_feat1_vals_nolab]
    teps_for_feat1vals = [[np.mean(results['test']['iteff_pred'][0,0,8:15,patient]) for patient in np.where(feat1_vals == featval)[0]] for featval in uniq_feat1_vals_nolab]
    stds_for_feat1vals = [[np.std(results['test']['iteff_pred'][0,0,8:15,patient]) for patient in np.where(feat1_vals == featval)[0]] for featval in uniq_feat1_vals_nolab]

    # for the visualisations of subgroup aggregates the means are pre-calculated
    treatment_means_1 = [np.mean(to_list) for to_list in treatment_outcomes_for_feat1vals]
    control_means_1 = [np.mean(co_list) for co_list in control_outcomes_for_feat1vals]

    # Feature 2/ Tumor Grade
    feat2_vals = results['test']['features'][0,0,0,:][:,feature2]
    uniq_feat2_vals = np.unique(feat2_vals)
    feat2val_labs = [str(featval) for featval in uniq_feat2_vals]
    treatment_outcomes_for_feat2vals = [[treatment_outcome_list[patient] for patient in np.where(feat2_vals == featval)[0]] for featval in uniq_feat2_vals]
    control_outcomes_for_feat2vals = [[control_outcome_list[patient] for patient in np.where(feat2_vals == featval)[0]] for featval in uniq_feat2_vals]
    teps_for_feat2vals = [[np.mean(results['test']['iteff_pred'][0,0,8:15,patient]) for patient in np.where(feat2_vals == featval)[0]] for featval in uniq_feat2_vals]
    treatment_means_2 = [np.mean(to_list) for to_list in treatment_outcomes_for_feat2vals]
    control_means_2 = [np.mean(co_list) for co_list in control_outcomes_for_feat2vals]


    # first generate 12 plots that do not involve individual patients
    # TODO: How these function are parameterised is certainly not optimal and could be reviewed
    p1_plot(treatment_outcome_list, control_outcome_list)
    p2_plot(teps_for_feat1vals, uniq_feat1_vals, treatment_outcome_list, control_outcome_list, feat1_vals)
    p3_plot(teps_for_feat1vals, stds_for_feat1vals, uniq_feat1_vals)
    p4_plot(avg_ite_per_person[0], treatment_outcome_list, control_outcome_list, feature1, feature2, results['test']['features'])
    p6_plot(results['test']['iteff_pred'], results['test']['features'], feature1, feature2)
    p7_plot(treatment_outcome_list, control_outcome_list, results['test']['features'], feature1, feature2)
    p8_plot(avg_ite_per_person[0], results['test']['features'], treatment_outcome_list, control_outcome_list, feature1)
    p10_plot(avg_ite_per_person[0], treatment_outcome_list, control_outcome_list)
    p12_plot(avg_ite_per_person)
    p14_plot(avg_ite_per_person, std_ite_per_person, treatment_outcome_list, control_outcome_list, results['test']['features'], feature1, feature2)
    p16_plot(avg_ite_per_person[0], std_ite_per_person[0])
    p17_plot(treatment_outcome_list, control_outcome_list, results['test']['features'], feature1, feature2)


    # then generate the 6 plots for individual patients Profile 18, 15, 5, 13, 9, 11
    for pat_nr in patient_nrs:
        ind_values_1 = [treatment_outcome_list[pat_nr], control_outcome_list[pat_nr]]
        print('Patient ' + str(pat_nr)+' Values: ', treatment_outcome_list[pat_nr], control_outcome_list[pat_nr])
        # execute plots for this person
        print('Plotting graphs for patient ' + str(pat_nr))
        p5_plot(treatment_outcome_list, control_outcome_list, feat1_vals, uniq_feat1_vals, pat_nr)
        p9_plot(avg_ite_per_person[0], std_ite_per_person[0], pat_nr)
        p11_plot(uniq_feat1_vals, uniq_feat2_vals, feat1_vals, feat2_vals, treatment_outcome_list, control_outcome_list, pat_nr, nr_patients)
        # for p13_plot read its description and readme!
        #p13_plot(avg_ite_per_person[0], treatment_outcome_list, control_outcome_list,results['test']['features'], results['test']['iteff_pred'], feature1, feature2, pat_nr)
        p15_plot(avg_ite_per_person[0], teps_for_feat1vals, feat1val_labs, pat_nr)
        p18_plot(avg_ite_per_person[0], treatment_outcome_list, control_outcome_list, pat_nr)


# p1_plot simply takes both prediction outcome lists and generates boxplots for each
def p1_plot(to_list, co_list):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.boxplot([to_list, co_list])
    ax.set_ylabel(["Treatment","Control"])
    plt.ylabel("Outcome Prediction")
    plt.savefig('p1.png', dpi=1000)
    plt.clf()
    plt.close()


# p2_plot sorts ites, treatment and control outcomes by the feature values of their respective patients and 
# samples a subset to keep the plot readable
def p2_plot(teps_for_featvals, all_featvals, to_list, co_list, feat_x):
    prc_to_sample = 0.06
    #teps_for_featvals = [[np.mean(iteff_pred_results[0,0,8:15,patient]) for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]

    featval_range = range(len(teps_for_featvals))

    # sample part of the indices for each unique feature expression because otherwise the plot gets too crowded
    featval_sizes = [len(teps_for_featvals[x]) for x in featval_range]
    sample_indices = [list(range(featval_sizes[x])) for x in featval_range] 
    [np.random.shuffle(sample_indices[x]) for x in featval_range]
    sampled_indices = [sample_indices[x][:math.floor(featval_sizes[x]*prc_to_sample)] for x in featval_range]

    
    tos_for_featvals = [[to_list[patient] for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]
    cos_for_featvals = [[co_list[patient] for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]
    sampled_tos = []
    samples_cos = []

    featval_samples = [random.sample(teps_for_featvals[x],math.floor(featval_sizes[x]*prc_to_sample)) for x in featval_range]

    tos_samples = [np.array(tos_for_featvals[x])[sampled_indices[x]] for x in featval_range]
    cos_samples = [np.array(cos_for_featvals[x])[sampled_indices[x]] for x in featval_range]
    ite_samples = [tos_samples[x]-cos_samples[x] for x in featval_range]

    nr_buckets = len(all_featvals) # nr_buckets is the number of unique feature expressions
    fig = plt.figure(figsize=(8,5))
    gs = fig.add_gridspec(1,nr_buckets, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    for ax in fig.get_axes():
        ax.label_outer()
    
    x_ticks_lists = [['',str(featval),''] for featval in all_featvals]
    x_ticks_list_t = ['','Treatment','']
    x_ticks_list_c = ['','Control','']

    sample_size = np.sum(featval_sizes)
    feat_names = ['70-74', '75-79', '80-84', '85+']

    # now for each unique feature expression sort the respective values and plot them as colored bars
    for ax_nr, ax_name in zip(range(nr_buckets), feat_names):
        # create a list of tuples to sort both lists by the size of the ite value
        to_sort = listOfTuples2(ite_samples[ax_nr],cos_samples[ax_nr])

        if(len(to_sort) < 50):
            sample_list = to_sort
        else:
            sample_list = random.sample(to_sort, 50)

        sorted_vals = sorted(sample_list, key=lambda tup: tup[0])
        sorted_ites = [x[0] for x in sorted_vals]
        sorted_cos = [x[1] for x in sorted_vals]

        # plot the ites and cos stacked on top of each other, depending on the ite being positive or negative
        for x in range(len(sorted_ites)-1):
            if(sorted_ites[x] > 0):
                axs[ax_nr].bar(x,sorted_cos[x],color='blue')
                axs[ax_nr].bar(x,sorted_ites[x], bottom = sorted_cos[x],color='red')
            else:
                axs[ax_nr].bar(x,sorted_ites[x],color='red')
                axs[ax_nr].bar(x,sorted_cos[x], bottom = 0 ,color='blue')

        # do this last step to add labels for the plot
        xx = len(sorted_ites) - 1 
        if(sorted_ites[x] > 0):
                axs[ax_nr].bar(xx,sorted_cos[xx],color='blue', label='Control Outcome')
                axs[ax_nr].bar(xx,sorted_ites[xx], bottom = sorted_cos[xx],color='red', label='Treatment Effect')
        else:
            axs[ax_nr].bar(xx,sorted_ites[xx],color='red', label='Treatment Effect')
            axs[ax_nr].bar(xx,sorted_cos[xx], bottom = 0 ,color='blue', label='Control Outcome')

        axs[ax_nr].set_xlabel(ax_name)

    axs[0].set_ylabel('Outcome Prediction')
    fig.supxlabel('Age Group')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('p2.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


# p3_plot samples part of the ite values for each expression of the target feature 
def p3_plot(teps_for_featvals, stds_for_featvals, all_featvals):
    #featval_sizes = [len(teps_for_featvals[x]) for x in range(len(teps_for_featvals))]

    #ordered_featvals = [np.sort([teps_for_featvals[x]]) for x in range(len(teps_for_featvals))]
    # o contains the same values as ordered featvals, it just removes one layer of unnecessary array structure
    #o = [x for xs in ordered_featvals for x in xs]

    nr_buckets = len(all_featvals)
    fig = plt.figure(figsize=(8,5))
    gs = fig.add_gridspec(1, nr_buckets, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    for ax in fig.get_axes():
        ax.label_outer()

    feat_names = ['70-74', '75-79', '80-84', '85+']
    color_list = ['red', 'blue', 'green', 'yellow']

    for colo, ax_nr, ax_name in zip(color_list, range(len(feat_names)), feat_names):
        to_sort = listOfTuples2(teps_for_featvals[ax_nr], stds_for_featvals[ax_nr])
        
        # before sorting, cull the list to a maximum of 50 values 
        if(len(to_sort) < 50):
            sample_list = to_sort
        else:
            sample_list = random.sample(to_sort, 50)
        
        sorted_vals = sorted(sample_list, key=lambda tup: tup[0])
        sorted_ites = [x[0] for x in sorted_vals]
        sorted_std = [x[1] for x in sorted_vals]

        for x, y ,std in zip(range(len(sorted_ites)), sorted_ites, sorted_std):
            axs[ax_nr].errorbar(x=x, y=y, yerr=std, fmt='o', ecolor='black', elinewidth=1, capsize=5, color='black')
        axs[ax_nr].set_xlabel(ax_name)

    fig.supxlabel('Age Group')
    axs[0].set_ylabel("Individual Treatment Effect")
    plt.savefig('p3.png', dpi=1000)
    plt.clf()
    plt.close()


# p4_plot visualises the averages for the unique feature expressions for 2 features over treatment and control outcome predictions, as well as
# the total averages over outcome predictions
def p4_plot(avg_ite_pp, to_list, co_list, feat1_nr, feat2_nr, feature_results):
    # feat_selection contains the indices for the given features in the dataset, TODO: maybe change to giving a list instead?
    feat_selection = [feat1_nr, feat2_nr]

    # calculate the total number of unique feature expressions 
    nr_categories = 0
    for f in feat_selection: 
        all_featvals = np.unique(feature_results[0,0,0,:][:,f])
        nr_categories = len(all_featvals) + nr_categories


    #ind = range(len(to_list))

    fig = plt.figure(figsize=(11, 5)) 
    gs = fig.add_gridspec(1,nr_categories, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    for ax in fig.get_axes():
        ax.label_outer()

    # these are used to add a dashed line with the average to and co outcomes over all predictions
    avg_treatment_outcome = np.mean(to_list)
    avg_control_outcome = np.mean(co_list)
    mean_points_treatment= np.array([avg_treatment_outcome, avg_treatment_outcome, avg_treatment_outcome])
    mean_points_control= np.array([avg_control_outcome, avg_control_outcome, avg_control_outcome])

    
    first = True
    start_ax = 0
    color_list = ('green','blue','blue')
    # TODO: features_names could be defined in prior
    feature_names = ['Age', 'Tumor Grade']
    features = feature_results[:,0,0,:][0]
    yval_labels = ['Treatment', 'Control']

    # the outer loop goes over all the features
    for feat, feat_name in zip(feat_selection, feature_names):
        # for the individual features, receive the mean values for each unique feature expression
        tos, cos, _, x_tick_labels = get_feature_split(features, feature_results, to_list, co_list, feature_nr=feat)

        # nr_featvals is used to determine where the plotting for the next feature has to end
        nr_featvals = len(tos) + start_ax

        # first add the dashed line, then the crosses for the prediction outcomes
        for ax_nr, feat_x in zip(range(start_ax, nr_featvals), range(len(tos))):
            # do this to not fill up legend
            if(first):
                axs[ax_nr].plot(mean_points_treatment, linestyle = 'dashed', color='green', label='Treatment Outcome Average')
                axs[ax_nr].plot(mean_points_control, linestyle = 'dashed', color='blue', label='Control Outcome Average')
                first = False
            else: 
                axs[ax_nr].plot(mean_points_treatment, linestyle = 'dashed', color='green')
                axs[ax_nr].plot(mean_points_control, linestyle = 'dashed', color='blue')

            yvals = [tos[feat_x], cos[feat_x]]
            ticklabels = x_tick_labels[feat_x]
            ticklabels[1] = feat_name + ' ' + ticklabels[1]

            for x, y, colors, lab in zip(range(nr_featvals), yvals, color_list, yval_labels):
                if(first):
                    axs[ax_nr].errorbar(x=x, y=y, fmt='x', ecolor='black', elinewidth=1, capsize=5, color = colors, label=lab)
                else:
                    axs[ax_nr].errorbar(x=x, y=y, fmt='x', ecolor='black', elinewidth=1, capsize=5, color = colors)
                axs[ax_nr].set_xticks(range(3))
                axs[ax_nr].set_xticklabels(ticklabels)
            
        # start_ax determines where the plotting for the next feature has to start from
        start_ax = len(tos) + start_ax

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].set_ylabel('Outcome Prediction')
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('p4.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


# p5_plot creates two partial dependence plots, one for treatment and one for control outcomes. 
# Points for a patients individual values are added
def p5_plot(to_list, co_list, feat_x, all_featvals, patient_nr):

    fig, ax = plt.subplots(figsize=(15, 10))

    patient_to = to_list[patient_nr]
    patient_co = co_list[patient_nr]
    # patient_feat marks, which unique feature expression the patient has
    patient_feat = feat_x[patient_nr]

    # both outcome prediction lists need to ne squeezed
    squeetreatment= np.squeeze(to_list)
    squeecontrol= np.squeeze(co_list)

    idx_list = []
    avg_pred_list_t = []
    avg_pred_list_c = []

    # calculate the averages over all unique feature expressions given in all_featvals
    for tup in all_featvals:
        first_val = tup
        # get the indices of patients that match the current feature expression
        idx_list = [i for i, e in enumerate(feat_x) if e == first_val]

        #need to work out what to do with 0 values
        treatment_pred_avg = np.mean([squeetreatment[i] for i in idx_list]) 
        control_pred_avg = np.mean([squeecontrol[i] for i in idx_list]) 

        # aggregate the means for each unique feature expression in a list for easy plotting
        avg_pred_list_t = avg_pred_list_t + [treatment_pred_avg]
        avg_pred_list_c = avg_pred_list_c + [control_pred_avg]


    plt.plot(all_featvals,avg_pred_list_t, color='green', label='Treatment')
    plt.plot(all_featvals,avg_pred_list_c, color='blue', label='Control')
    plt.errorbar(x=patient_feat, y=patient_to, fmt='o', ecolor='black', elinewidth=1, capsize=5, color = 'green', label="Patient Treatment Outcome ")
    plt.errorbar(x=patient_feat, y=patient_co, fmt='o', ecolor='black', elinewidth=1, capsize=5, color = 'blue', label="Patient Control Outcome ")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Patient Age')
    plt.ylabel('Outcome Prediction')
    plt.savefig('p5_pat'+ str(patient_nr) + '.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


# p6_plot is based on forest plots/hazard ratios. It aggregates the ite values for the unique feature expressions and puts them in a zepid plot
def p6_plot(iteff_pred_results, feature_results, feat1_nr, feat2_nr):
    feat_list = [feat1_nr, feat2_nr]
    teps_for_featvals = []
    labs = []

    feat_names = ['Age ', 'Tumor Grade ']
    feat_group_names = ['70-74', '75-79', '80-84', '85+']

    # choose labels according to feature names
    for f, name in zip(feat_list, feat_names):
        feat_x = feature_results[0,0,0,:][:,f]
        all_featvals = np.unique(feat_x)

        # look at all patients that share a feature expression (featval) and take the mean over their ites at different points in training
        teps_for_featvals = teps_for_featvals + [[np.mean(iteff_pred_results[0,0,8:15,patient]) for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]

        if(f == 0):
            all_featvals = feat_group_names
        else:
            all_featvals = np.unique(feat_x)
        labs = labs + [name + str(featval) for featval in all_featvals]
    

    ordered_featvals = [np.sort([teps_for_featvals[x]]) for x in range(len(teps_for_featvals))]
    # same purpose as in p3_plot, just gets rid of one extra array layer
    o = [x for xs in ordered_featvals for x in xs]

    measure = [round(np.mean(vals),1) for vals in o]
    std = [round(np.std(vals),1) for vals in o]
    lower = [round(measure[x] - std[x],1) for x in range(len(measure))]
    upper = [round(measure[x] + std[x],1) for x in range(len(measure))]
    upper_bound = round((np.max(upper) + 1), 1)
    lower_bound = round((np.min(lower) - 1), 1)
    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(effectmeasure='Avg ITE', conf_int='One Std', center= 0)
    p.colors(pointshape="x")
    ax=p.plot(figsize=(8,5), t_adjuster=0.09, max_value=upper_bound, min_value=lower_bound )
    plt.title("Individual Treatment Effects = ITE",loc="right",x=0, y=1.045)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

    plt.savefig("p6.png", bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


# p7_plot 
def p7_plot(to_list, co_list, feature_results, feat1_nr, feat2_nr):


    nr_categories = 4 

    fig = plt.figure()
    gs = fig.add_gridspec(1,nr_categories, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    for ax in fig.get_axes():
        ax.label_outer()
    features = feature_results[:,0,0,:][0]
    
    # create these arrays of 3 values to fit into the 3 xticks used in every subplots
    avg_treatment_outcome = np.mean(to_list)
    avg_control_outcome = np.mean(co_list)
    mean_points_treatment= np.array([avg_treatment_outcome, avg_treatment_outcome, avg_treatment_outcome])
    mean_points_control= np.array([avg_control_outcome, avg_control_outcome, avg_control_outcome])

    # define parameters that are used for plotting
    feat_selection = [feat1_nr]
    labs = ['Treatment', 'Control']
    ind = range(len(to_list))
    color_list = ('green','blue','blue')

    for feat in feat_selection:
        # for the individual features, receive the mean values for each unique feature expression
        tos, cos, _, x_tick_labels = get_feature_split(features, feature_results, to_list, co_list, feature_nr=feat)

        nr_featvals = len(tos)
        for ax_nr in range(nr_featvals):
            yvals = [tos[ax_nr], cos[ax_nr]]
            ticklabels = x_tick_labels[ax_nr]
            axs[ax_nr].plot(mean_points_treatment, linestyle = 'dashed', color='green', label= 'Average Treatment')
            axs[ax_nr].plot(mean_points_control, linestyle = 'dashed', color='blue', label= 'Average Control')
            axs[ax_nr].set_xticks(range(3))
            axs[ax_nr].set_xticklabels(ticklabels)
            for x, y, colors, l in zip(range(nr_featvals), yvals, color_list, labs):
                axs[ax_nr].errorbar(x=x, y=y, fmt='x', ecolor='black', elinewidth=1, capsize=5, color = colors, label=l)
                

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].set_ylabel('Outcome Prediction')
    fig.supxlabel('Age Group')
    plt.savefig('p7.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


# p8_plot is a simple one-way pdp over a single feature and ites
def p8_plot(avg_ite, feature_results, to_list, co_list, feature_nr):

    fig, ax = plt.subplots(figsize=(10, 5))

    feat_x = feature_results[0,0,0,:][:,feature_nr]
    all_featvals = np.unique(feat_x)
    first_feat = feat_x
    squeeite= np.squeeze(avg_ite)

    # TODO: does initialising make sense here?
    idx_list = []
    avg_pred_list = []

    for tup in all_featvals:
        first_val = tup
        idx_list = [i for i, e in enumerate(first_feat) if e == first_val]

        #need to work out what to do with 0 values
        ite_pred_avg = np.mean([squeeite[i] for i in idx_list]) 
        avg_pred_list = avg_pred_list + [ite_pred_avg]

    plt.plot(all_featvals,avg_pred_list, color='black', label='ITE')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Patient Age')
    ax.set_ylabel('Individual Treatment Effect')
    plt.savefig('p8.png', dpi=1000)
    plt.clf()


# p9_plot shows ites for a sample of the population, depicted as points with whiskers for standard deviation
def p9_plot(avg_ite, std_ite, patient_nr):

    # have the patient to be marked in red
    color_list = ['black' if x!=patient_nr else 'red' for x in range(len(avg_ite))]
    ite_with_stds = listOfTuples3(avg_ite,std_ite, color_list)

    # sample a number of patients to not overcrowd the plot, then add the patient to be shown
    patient_tup = ite_with_stds[patient_nr]
    sample_list = random.sample(ite_with_stds, 75)
    complete_sample_list = sample_list.append(patient_tup)
    sorted_ite_with_stds = sorted(sample_list, key=lambda tup: tup[0])

    
    sorted_ites = [x[0] for x in sorted_ite_with_stds]
    sorted_stds = [x[1] for x in sorted_ite_with_stds]
    sorted_colors= [x[2] for x in sorted_ite_with_stds]

    fig, ax = plt.subplots(figsize=(10, 5))
    for x, y, err, colors in zip(range(len(sorted_ites)), sorted_ites, sorted_stds, sorted_colors):
        ax.errorbar(x=x, y=y, yerr=err, fmt='o', ecolor='black', elinewidth=1, capsize=5, color = colors)

    plt.ylabel('Individual Treatment Effect')
    plt.xlabel('Individual Patients')
    plt.savefig('p9_pat' + str(patient_nr)+ '.png', dpi=1000)
    plt.clf()
    plt.close()


# 
def p10_plot(avg_ite, to_list, co_list):
    fig, ax = plt.subplots(figsize=(10, 5))

    # create tuples of ites, cos and tos for sorting and sampling
    ite_with_stds = listOfTuples3(avg_ite,to_list, co_list)

    sample_list = random.sample(ite_with_stds, 75)

    sorted_ite_with_stds = sorted(sample_list, key=lambda tup: tup[0])
    sorted_ites = [x[0] for x in sorted_ite_with_stds]
    sorted_treatments = [x[1] for x in sorted_ite_with_stds]
    sorted_controls= [x[2] for x in sorted_ite_with_stds]


    #sorted_treatment_effect_list = [sorted_treatments[x]-sorted_controls[x] for x in range(len(sorted_treatments))]
    
    # first plot the negative ites without control values
    negative_ite = [x for x in sorted_ites if x<0]
    negative_inds = range(len(negative_ite))
    ax.bar(negative_inds, negative_ite, color='r')

    # then plot all the control values and only the positive ites
    positive_ite = [x if x>0 else 0 for x in sorted_ites]
    ind = range(len(sorted_treatments))
    ax.bar(ind,sorted_controls, color='b', label='Control Outcomes')
    ax.bar(ind,positive_ite, bottom=sorted_controls, color='r', label='Treatment Effect')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Individual Patients")
    plt.ylabel("Outcome Prediction")
    plt.savefig('p10.png' , bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


# p11_plot is special as the feature groups are not taken in separation from each other but groups defined by two feature expression are constructed
def p11_plot(all_feat1vals, all_feat2vals, feat_1, feat_2, to_list, co_list, patient_nr, nr_patients):

    nr1_buckets = len(all_feat1vals)
    nr2_buckets = len(all_feat2vals)

    # get indices for the subgroups, to match and later extract the outcome predictions
    teps_for_feat1vals = [np.where(feat_1 == featval)[0] for featval in all_feat1vals]
    teps_for_feat2vals = [np.where(feat_2 == featval)[0] for featval in all_feat2vals]
    indices_for_feat_combs = [[np.intersect1d(ind1,ind2) for ind2 in teps_for_feat2vals] for ind1 in teps_for_feat1vals]

    # get individual treatment and control outcomes for all feature combinations each is #different fea1vals x # different feat2val 
    treatment_outcomes_for_feat_combs = [[[to_list[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]
    control_outcomes_for_feat_combs = [[[co_list[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]

    # first create a color list as usual, then split the list for the different feature combinations.
    color_list = ['red' if x!=patient_nr else 'black' for x in range(nr_patients)]
    color_list_control = ['blue' if x!=patient_nr else 'yellow' for x in range(nr_patients)]
    colors_ite = [[[color_list[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]
    colors_control = [[[color_list_control[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]

    fig = plt.figure(figsize = (10, 8))
    gs = fig.add_gridspec(nr1_buckets,nr2_buckets, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    for ax in fig.get_axes():
        ax.label_outer()


    feat_names = ['70-74', '75-79', '80-84', '85+']
    y_ticks_lists = [str(featval) for featval in feat_names]
    y_ticks_lists.reverse()
    x_ticks_lists = [str(featval) for featval in all_feat2vals]

    first=True

    patient_tup = (to_list[patient_nr],co_list[patient_nr],'black','yellow')

    # 
    for ax1_nr in range(nr1_buckets):
        axs[3,ax1_nr].set_xlabel(x_ticks_lists[ax1_nr])

        for ax2_nr in range(nr2_buckets):

            axs[ax2_nr,0].set_ylabel(y_ticks_lists[ax2_nr])     
            # create a tuple list which can be sampled and then sorted for any combination of unique feature expressions
            ite_with_stds = listOfTuples4(treatment_outcomes_for_feat_combs[ax1_nr][ax2_nr], control_outcomes_for_feat_combs[ax1_nr][ax2_nr], \
             colors_ite[ax1_nr][ax2_nr], colors_control[ax1_nr][ax2_nr])

            if(len(ite_with_stds) < 50):
                sample_list = ite_with_stds
            else:
                sample_list = random.sample(ite_with_stds, 50)

            # test if the patient to be marked is already included in the sample_list, otherwise add it
            if(all_feat1vals[ax1_nr] == feat_1[patient_nr] and all_feat2vals[ax2_nr] == feat_2[patient_nr] and (patient_tup not in sample_list)):
                sample_list.append(patient_tup)

            sorted_ite_with_stds = sorted(sample_list, key=lambda tup: tup[0]-tup[1])
            sorted_treatments = [x[0] for x in sorted_ite_with_stds]
            sorted_controls= [x[1] for x in sorted_ite_with_stds]
            sorted_colors_ite = [x[2] for x in sorted_ite_with_stds]
            sorted_color_control = [x[3] for x in sorted_ite_with_stds]

            sorted_treatment_effect_list = [sorted_treatments[x]-sorted_controls[x] for x in range(len(sorted_treatments))]
            ind = range(len(sorted_treatments))


            for x, sc, ite, colo, ccolor in zip(ind, sorted_controls, sorted_treatment_effect_list, sorted_colors_ite, sorted_color_control):

                if(ite > 0):
                    # do this so the legend does not spill over
                    if(first==True):
                        axs[ax1_nr,ax2_nr].bar(x,sc,color=ccolor, label='Control Outcome')
                        axs[ax1_nr,ax2_nr].bar(x,ite, bottom=sc, color=colo, label = 'Treatment Effect')
                        first = False
                    else:
                        axs[ax1_nr,ax2_nr].bar(x,sc,color=ccolor)
                        axs[ax1_nr,ax2_nr].bar(x,ite, bottom=sc, color=colo)
                else:
                    axs[ax1_nr, ax2_nr].bar(x,ite,color=colo)
                    axs[ax1_nr, ax2_nr].bar(x,sc,color=ccolor)

    fig.supylabel('Age Group')
    fig.supxlabel('Tumor Grade')
    plt.savefig('p11_pat' + str(patient_nr)+'.png', dpi=1000, bbox_inches= 'tight')
    plt.clf()
    plt.close()


# p12_plot is just a simple boxplot of ites
def p12_plot(avg_ite):
    fig = plt.figure()

    avg = list(list(avg_ite)[0])

    for ax in fig.get_axes():
        ax.label_outer()

    plt.boxplot(avg)
    plt.ylabel('Individual Treatment Effect')
    plt.savefig('p12.png', dpi=1000)
    plt.clf()
    plt.close()


# p13_plot creates a two way partial dependence plot. Pay attention to replace the code from sklearn as specified in the README!
def p13_plot(avg_ite, to_list, co_list,feat_results, iteff_pred_results, feat1_nr, feat2_nr, patient_nr):

    fig, ax = plt.subplots()

    # these serve to prepare the dummy estimator
    big_X = np.squeeze(np.mean(feat_results[:,0,:,:], axis=(1)))
    true_Y = np.squeeze(np.mean(iteff_pred_results[:,0,:,:], axis=(1)))

    # needs to be squeezed, otherwise this is interpreted as multi-output regressor
    squeeite= np.squeeze(avg_ite)

    dum = DummyRegressor(wanted_output=squeeite)
    dum.fit(big_X,true_Y)

    # tep_results are not native to from_estimator. Due to the uncommon structure of the cfrnet i have to pass the results and work with them
    pdp_plot = PartialDependenceDisplay.from_estimator(
        estimator=dum,
        X=big_X,
        features=[(feat1_nr,feat2_nr)], # the features to plot
        tep_results=iteff_pred_results[:,0,:,:],
        method='brute',
        ax=ax,
    )

    # now plot two lines marking the individual patients feature expression onto the two way pdp
    feat_1 = feat_results[0,0,0,:][:,feat1_nr]
    feat_2 = feat_results[0,0,0,:][:,feat2_nr]

    patient_f1val = feat_1[patient_nr]
    patient_f2val = feat_2[patient_nr]
    # add a little offset so that the line does not disappear at the left side of the plot
    if(patient_f1val == 70):
        patient_f1val = patient_f1val+2
    plt.axvline(patient_f1val, label='Patient Features', color='black')

    plt.axhline(patient_f2val, color='black')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Patient Age')
    plt.ylabel('Tumor Grade')
    plt.savefig('p13_pat' + str(patient_nr) + '.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


# p14_plot like p11 creates subgroups of patients that share 2 feature expression of 2 features
def p14_plot(avg_ite, std_ite, to_list, co_list, feature_results, feat1_nr, feat2_nr):

    prc_to_sample = 0.06

    feat_1 = feature_results[0,0,0,:][:,feat1_nr]
    feat_2 = feature_results[0,0,0,:][:,feat2_nr]
    all_feat1vals = np.unique(feat_1)
    all_feat2vals = np.unique(feat_2)

    # get indeces for the subgroups, to match and later extract the outcome predictions
    tep_indices_for_feat1vals = [np.where(feat_1 == featval)[0] for featval in all_feat1vals]
    tep_indices_for_feat2vals = [np.where(feat_2 == featval)[0] for featval in all_feat2vals]

    indices_for_feat_combs = [[np.intersect1d(ind1,ind2) for ind2 in tep_indices_for_feat2vals] for ind1 in tep_indices_for_feat1vals]

    avg_ite_per_person = list(list(avg_ite)[0])
    std_ite_per_person = list(list(std_ite)[0])

    # aggregate ites and standard deviations according to the characteristics of the respective patients
    ite_preds_for_feat_combs = [[[avg_ite_per_person[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]
    std_preds_for_feat_combs = [[[std_ite_per_person[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]

    # TODO: is this necessary? or could i use the above shape?
    nr1_buckets = len(all_feat1vals)
    nr2_buckets = len(all_feat2vals)

    fig = plt.figure(figsize=(15,10))
    gs = fig.add_gridspec(nr1_buckets,nr2_buckets, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    for ax in fig.get_axes():
        ax.label_outer()
    
    feat_names = ['70-74', '75-79', '80-84', '85+']
    y_ticks_lists = [str(featval) for featval in feat_names]
    y_ticks_lists.reverse()
    x_ticks_lists = [str(featval) for featval in all_feat2vals]

    for ax1_nr in range(nr1_buckets):
        axs[3,ax1_nr].set_xlabel(x_ticks_lists[ax1_nr])

        for ax2_nr in range(nr2_buckets):
            axs[ax2_nr,0].set_ylabel(y_ticks_lists[ax2_nr])

            to_sort = listOfTuples2(ite_preds_for_feat_combs[ax1_nr][ax2_nr], std_preds_for_feat_combs[ax1_nr][ax2_nr])
            if (len(to_sort) < 50):
                sample_list = random.sample(to_sort, len(to_sort))
            else:
                sample_list = random.sample(to_sort, 50)

            sorted_ite_with_stds = sorted(sample_list, key=lambda tup: tup[0])
            sorted_treatment_effect_list = [x[0] for x in sorted_ite_with_stds]
            sorted_stds = [x[1] for x in sorted_ite_with_stds]


            for x, y, std in zip(range(len(sorted_treatment_effect_list)), sorted_treatment_effect_list, sorted_stds):
                axs[ax1_nr,ax2_nr].errorbar(x=x, y=y, yerr=std, fmt='o', ecolor='black', elinewidth=1, capsize=5, color = 'black')


    fig.supylabel('Age Group')
    fig.supxlabel('Tumor Grade')
    plt.savefig('p14.png', dpi=1000)
    plt.clf()
    plt.close()


# p15_plot creates a hazard ratio/forest plot that shows ites aggregated over groups defined by shared feature expressions
def p15_plot(avg_ite, teps_for_featval, featval_labs, patient_nr):
    # sort the ite values that are grouped by shared feature expression TODO: maybe doing the sorting once for all profiles could be useful!
    ordered_featvals = [np.sort([teps_for_featval[x]]) for x in range(len(teps_for_featval))]
    # this is done to correct the shape/ remove an array dimension or something
    o = [x for xs in ordered_featvals for x in xs]

    feat_names = ['Age 70-74', 'Age 75-79', 'Age 80-84', 'Age 85+']

    patient_ite = avg_ite[patient_nr]
    measure = [round(np.mean(vals),1) for vals in o]
    std = [round(np.std(vals),1) for vals in o]
    lower = [round(measure[x] - std[x],1) for x in range(len(measure))]
    upper = [round(measure[x] + std[x],1) for x in range(len(measure))]
    upper_bound = round((np.max(upper) + 1), 1)
    lower_bound = round((np.min(lower) - 1), 1)
    p = EffectMeasurePlot(label=feat_names, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(effectmeasure='Avg ITE', conf_int='One Std', center=0)
    p.colors(pointshape="x")
    
    ax=p.plot(figsize=(8,5), t_adjuster=0.09, max_value=upper_bound, min_value=lower_bound )
    ax.axvline(patient_ite, color='red', label='Patient ITE')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.legend()

    plt.savefig("p15_pat" + str(patient_nr) + ".png", bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


# p16_plot creates a simple histogram that aggregates ite values
def p16_plot(avg_ite, std_ite):
    ite_with_stds = listOfTuples2(avg_ite, std_ite)
    sorted_ite_with_stds = sorted(ite_with_stds, key=lambda tup: tup[0])
    sorted_ites = [x[0] for x in sorted_ite_with_stds]

    # sort ites in buckets of width 1
    fig = plt.figure()
    max_ite = int(max(avg_ite)) + 2  # +1 somehow does not create the last bucket at the top
    min_ite = int(min(avg_ite)) -1
    
    bins = np.array(range(min_ite,max_ite))
    ites_in_bins = pd.cut(avg_ite,bins)
    ite_hist = np.histogram(avg_ite,bins)

    plt.hist(avg_ite,list(ite_hist[1]), color= 'black')
    plt.xlabel('Individual Treatment Effect')
    plt.ylabel('Number of Patients')
    plt.savefig('p16.png', dpi=1000)
    plt.clf()
    plt.close()


# p17_plot creates pdps for the different features over all feature expressions
def p17_plot(to_list, co_list, feature_results, feat1_nr, feat2_nr):
    fig, ax = plt.subplots(figsize=(15, 10))

    squeetreatment= np.squeeze(to_list)
    squeecontrol= np.squeeze(co_list)
    feat_list = [feat1_nr,feat2_nr]
    names_list = ['Age', 'Tumor Grade']
    colors = [['green', 'blue'], ['yellow', 'red']]

    for feat_nr, feat_name, col in zip(feat_list,names_list, colors):
        first_feat = feature_results[0,0,0,:][:,feat_nr]
        all_featvals = np.unique(first_feat)

        # fv_indices are the index values for any given feature value
        idx_list = []
        avg_pred_list_t = []
        avg_pred_list_c = []
        for tup in all_featvals:
            first_val = tup
            fv_indices = [i for i, e in enumerate(first_feat) if e == first_val]

            patient_idx = [value for value in fv_indices]
            idx_list = idx_list + patient_idx

            #need to work out what to do with 0 values
            treatment_pred_avg = np.mean([squeetreatment[i] for i in idx_list]) 
            control_pred_avg = np.mean([squeecontrol[i] for i in idx_list]) 

            avg_pred_list_t = avg_pred_list_t + [treatment_pred_avg]
            avg_pred_list_c = avg_pred_list_c + [control_pred_avg]
            idx_list = []


        avg_pred_list_t = np.reshape(avg_pred_list_t, (len(all_featvals)))
        avg_pred_list_c = np.reshape(avg_pred_list_c, (len(all_featvals)))

        t_label = feat_name + ' Treatment'
        c_label = feat_name + ' Control'
        plt.plot(all_featvals,avg_pred_list_t, color=col[0], label=t_label)
        plt.plot(all_featvals,avg_pred_list_c, color=col[1], label=c_label)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Feauture Value')
    plt.ylabel('Outcome Predictions')

    plt.savefig('p17.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()


# p18_plot creates a joint histogramm of tos and pos. It also adds marks for the outcome predictions of a single patient
def p18_plot(avg_ite, t_outcomes, c_outcomes, patient_nr):
    chosen_patient_treatment_outcome = t_outcomes[patient_nr]
    chosen_patient_control_outcome = c_outcomes[patient_nr]

    min_treatment = int(min(t_outcomes))   
    max_treatment = int(max(t_outcomes)) 
    min_control = int(min(c_outcomes))   
    max_control = int(max(c_outcomes)) 

    # need bins for min control and max treatment supposedly
    bin_nr = max_treatment - min_control

    hist_values = np.array(list(zip(t_outcomes, c_outcomes)))
    plt.hist(hist_values,bin_nr, label=['Treatment', 'Control'], color=['green', 'blue']) # density=True, histtype='bar',

    plt.axvline(chosen_patient_treatment_outcome, label="Patient predicted Treatment Outcome", color='orange')
    plt.axvline(chosen_patient_control_outcome, label="Patient predicted Control Outcome", color='yellow')

    plt.xlabel('Outcome Prediction')
    plt.ylabel('Number of Patients')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('p18_pat' + str(patient_nr) +'.png', bbox_inches='tight', dpi=1000)
    plt.clf()
    plt.close()

