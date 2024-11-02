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
    feat1_vals = results['test']['features'][0,0,0,:][:,feature1]
    uniq_feat1_vals_nolab = np.unique(feat1_vals) 
    uniq_feat1_vals = np.unique(feat1_vals)  # ['70-74', '75-79', '80-84', '85+']
    feat1val_labs = [str(featval) for featval in uniq_feat1_vals_nolab]
    treatment_outcomes_for_feat1vals = [[treatment_outcome_list[patient] for patient in np.where(feat1_vals == featval)[0]] for featval in uniq_feat1_vals_nolab]
    control_outcomes_for_feat1vals = [[control_outcome_list[patient] for patient in np.where(feat1_vals == featval)[0]] for featval in uniq_feat1_vals_nolab]
    teps_for_feat1vals = [[np.mean(results['test']['iteff_pred'][0,0,8:15,patient]) for patient in np.where(feat1_vals == featval)[0]] for featval in uniq_feat1_vals_nolab]
    stds_for_feat1vals = [[np.std(results['test']['iteff_pred'][0,0,8:15,patient]) for patient in np.where(feat1_vals == featval)[0]] for featval in uniq_feat1_vals_nolab]
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
    p1_plot(treatment_outcome_list, control_outcome_list)
    p2_plot(results['test']['iteff_pred'], uniq_feat1_vals, treatment_outcome_list, control_outcome_list, feat1_vals)
    p3_plot(teps_for_feat1vals, stds_for_feat1vals, uniq_feat1_vals)
    p4_plot(avg_ite_per_person[0], treatment_outcome_list, control_outcome_list, feature1, feature2, results['test']['features'])
    p6_plot(results['test']['iteff_pred'], results['test']['features'], feature1, feature2)
    p7_plot(avg_ite_per_person[0], treatment_outcome_list, control_outcome_list, results['test']['features'], feature1, feature2)
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
        p13_plot(avg_ite_per_person[0], treatment_outcome_list, control_outcome_list,results['test']['features'], results['test']['iteff_pred'], feature1, feature2, pat_nr)
        p15_plot(avg_ite_per_person[0], teps_for_feat1vals, feat1val_labs, pat_nr)
        p18_plot(avg_ite_per_person[0], treatment_outcome_list, control_outcome_list, pat_nr)



def p1_plot(to_list, co_list):
    fig, ax = plt.subplots(figsize=(10, 5))
    to_mean = np.mean(to_list)
    co_mean = np.mean(co_list)
    subtext = ("Treatment","Control")
    plt.boxplot([to_list, co_list])
    ax.set_ylabel(["Treatment","Control"])
    plt.ylabel("Outcome Prediction")
    plt.savefig('p1.png', dpi=1000)
    plt.clf()


def p2_plot(iteff_pred_results, all_featvals, to_list, co_list, feat_x):
    prc_to_sample = 0.06
    teps_for_featvals = [[np.mean(iteff_pred_results[0,0,8:15,patient]) for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]

    nr_buckets = len(all_featvals)

    tos_for_featvals = [[to_list[patient] for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]
    cos_for_featvals = [[co_list[patient] for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]

    featval_sizes = [len(teps_for_featvals[x]) for x in range(len(teps_for_featvals))]
    sample_indices = [list(range(featval_sizes[x])) for x in range(len(all_featvals))] #[:math.floor(featval_sizes[x]*prc_to_sample)]
    [np.random.shuffle(sample_indices[x]) for x in range(len(all_featvals))]
    sampled_indices = [sample_indices[x][:math.floor(featval_sizes[x]*prc_to_sample)] for x in range(len(all_featvals))]

    #np.random.shuffle(sample_indices)

    sampled_tos = []
    samples_cos = []

    featval_samples = [random.sample(teps_for_featvals[x],math.floor(featval_sizes[x]*prc_to_sample)) for x in range(len(all_featvals))]

    tos_samples = [np.array(tos_for_featvals[x])[sampled_indices[x]] for x in range(len(all_featvals))]
    cos_samples = [np.array(cos_for_featvals[x])[sampled_indices[x]] for x in range(len(all_featvals))]

    ite_samples = [tos_samples[x]-cos_samples[x] for x in range(len(all_featvals))]
    fig = plt.figure(figsize=(8,5))
    gs = fig.add_gridspec(1,nr_buckets, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    #fig.suptitle('Profile 2: Control and Treatment Outcomes for feature groups')

    for ax in fig.get_axes():
        ax.label_outer()
    
    x_ticks_lists = [['',str(featval),''] for featval in all_featvals]
    x_ticks_list_t = ['','Treatment','']
    x_ticks_list_c = ['','Control','']

    sample_size = np.sum(featval_sizes)
    feat_names = ['70-74', '75-79', '80-84', '85+']

    for ax_nr, ax_name in zip(range(nr_buckets), feat_names):
        to_sort = listOfTuples2(ite_samples[ax_nr],cos_samples[ax_nr])
        if(len(to_sort) < 50):
            sample_list = to_sort
        else:
            sample_list = random.sample(to_sort, 50)
        sorted_vals = sorted(sample_list, key=lambda tup: tup[0])
        sorted_ites = [x[0] for x in sorted_vals]
        sorted_cos = [x[1] for x in sorted_vals]
        for x in range(len(sorted_ites)-1):
            if(sorted_ites[x] > 0):
                axs[ax_nr].bar(x,sorted_cos[x],color='blue')
                axs[ax_nr].bar(x,sorted_ites[x], bottom = sorted_cos[x],color='red')
            else:
                axs[ax_nr].bar(x,sorted_ites[x],color='red')
                axs[ax_nr].bar(x,sorted_cos[x], bottom = 0 ,color='blue')

        xx = len(sorted_ites) - 1 
        if(sorted_ites[x] > 0):
                axs[ax_nr].bar(xx,sorted_cos[xx],color='blue', label='Control Outcome')
                axs[ax_nr].bar(xx,sorted_ites[xx], bottom = sorted_cos[xx],color='red', label='Treatment Effect')
        else:
            axs[ax_nr].bar(xx,sorted_ites[xx],color='red', label='Treatment Effect')
            axs[ax_nr].bar(xx,sorted_cos[xx], bottom = 0 ,color='blue', label='Control Outcome')

        axs[ax_nr].set_xlabel(ax_name)

    axs[0].set_ylabel('Outcome Prediction')
    #plt.xlabel("Patient Age")
    fig.supxlabel('Age Group')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('p2.png', bbox_inches='tight', dpi=1000)
    plt.clf()


def p3_plot(teps_for_featvals, stds_for_featvals, all_featvals):
    featval_sizes = [len(teps_for_featvals[x]) for x in range(len(teps_for_featvals))]

    nr_buckets = len(all_featvals)

    # 2. of each feature, sample a random subset of relative size. This size needs to be chosen manualy so that the individuals can be made out
    # featval_samples = [random.sample(teps_for_featvals[x],math.floor(featval_sizes[x]*prc_to_sample)) for x in range(len(all_featvals))]

    # 3. maybe order the feature subsets 
    ordered_featvals = [np.sort([teps_for_featvals[x]]) for x in range(len(teps_for_featvals))]
    # how do i get this in one line?
    o = [x for xs in ordered_featvals for x in xs]
    #ordered_featval_list = [x for xs in o for x in xs]

    # 4. visualize
    fig = plt.figure(figsize=(8,5))
    gs = fig.add_gridspec(1, nr_buckets, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    #fig.suptitle('Sorted ITEs vor variants of one feature')

    for ax in fig.get_axes():
        ax.label_outer()

    feat_names = ['70-74', '75-79', '80-84', '85+']

    sample_size = np.sum(featval_sizes)
    #sorted_treatment_effect_list = [sorted_treatments[x]-sorted_controls[x] for x in range(len(treatment_outcome_list))]
    #ind = range(len(ordered_featval_list))
    #color_list = ['red' if x!=patient_nr else 'black' for x in range(len(ordered_featval_list))]
    color_list = ['red', 'blue', 'green', 'yellow']

    startint = 0 
    for ite, colo, ax_nr, ax_name in zip(o, color_list, range(len(feat_names)), feat_names):
        to_sort = listOfTuples2(teps_for_featvals[ax_nr], stds_for_featvals[ax_nr])
        #ax_len = len(teps_for_featvals[ax_nr])
        
        if(len(to_sort) < 50):
            sample_list = to_sort
        else:
            sample_list = random.sample(to_sort, 50)
        sorted_vals = sorted(sample_list, key=lambda tup: tup[0])
        sorted_ites = [x[0] for x in sorted_vals]
        sorted_std = [x[1] for x in sorted_vals]
        #lbl = 'Age ' + str(ax_name)
        for x, y ,std in zip(range(len(sorted_ites)), sorted_ites, sorted_std):
            axs[ax_nr].errorbar(x=x, y=y, yerr=std, fmt='o', ecolor='black', elinewidth=1, capsize=5, color='black')
        axs[ax_nr].set_xlabel(ax_name)

    #ax.legend()
    #plt.xlabel("Age Group")
    fig.supxlabel('Age Group')
    axs[0].set_ylabel("Individual Treatment Effect")
    #plt.title("Sorted ITEs for variants of one feature")
    plt.savefig('p3.png', dpi=1000)
    plt.clf()


def p4_plot(avg_ite_pp, to_list, co_list, feat1_nr, feat2_nr, feature_results):
    features = feature_results[:,0,0,:][0]
    feat_selection = [feat1_nr, feat2_nr]

    # here i could exchange the mean_..._total values with the mean TEP value 
    #point_values = [mean_feature6_total, mean_treatment_outcomes, mean_control_outcomes, mean_invert_feature6_total, mean_invert_treatment_outcomes, mean_invert_control_outcomes]

    nr_bucks = 0
    for f in feat_selection: 
        feat_x = feature_results[0,0,0,:][:,f]
        all_featvals = np.unique(feat_x)
        nr_bucks = len(all_featvals) + nr_bucks


    nr_categories = nr_bucks 
    ind = range(len(to_list))
    color_list = ('green','blue','blue')
    #color_list = colors*nr_categories

    fig = plt.figure(figsize=(11, 5)) #figsize=(16, 8))
    gs = fig.add_gridspec(1,nr_categories, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    #fig.suptitle('Profile 4: Control and Treatment Outcomes for different feature groups')

    for ax in fig.get_axes():
        ax.label_outer()


    #tos, cos, ind = get_feature_split(features, feature_nr=0)

    avg_ite = np.mean(avg_ite_pp)
    avg_treatment_outcome = np.mean(to_list)
    avg_control_outcome = np.mean(co_list)

    mean_points_ite = np.array([avg_ite, avg_ite, avg_ite])
    mean_points_treatment= np.array([avg_treatment_outcome, avg_treatment_outcome, avg_treatment_outcome])
    mean_points_control= np.array([avg_control_outcome, avg_control_outcome, avg_control_outcome])


    # do this to not fill up legend
    first = True
    start_ax = 0
    feature_names = ['Age', 'Tumor Grade']
    
    #plt.xticks(range(6),x_ticks_list)
    for feat, feat_name in zip(feat_selection, feature_names):
        tos, cos, ind, x_tick_labels = get_feature_split(features, feature_results, to_list, co_list, feature_nr=feat)

        #treatment_ind = np.array([ind[0], ind[0], ind[0]])
        #control_ind = np.array([ind[1], ind[1], ind[1]])
        nr_featvals = len(tos) + start_ax
        for ax_nr, feat_x in zip(range(start_ax, nr_featvals), range(len(tos))):
            #yt, yo, yi = tos[ax_nr], cos[ax_nr], ind=[ax_nr]
            yvals = [tos[feat_x], cos[feat_x]]
            yval_labels = ['Treatment', 'Control']
            ticklabels = x_tick_labels[feat_x]
            ticklabels[1] = feat_name + ' ' + ticklabels[1]

            if(first):
                axs[ax_nr].plot(mean_points_treatment, linestyle = 'dashed', color='green', label='Treatment Outcome Average')
                axs[ax_nr].plot(mean_points_control, linestyle = 'dashed', color='blue', label='Control Outcome Average')
                
            else: 
                axs[ax_nr].plot(mean_points_treatment, linestyle = 'dashed', color='green')
                axs[ax_nr].plot(mean_points_control, linestyle = 'dashed', color='blue')

            for x, y, colors, lab in zip(range(nr_featvals), yvals, color_list, yval_labels):
                if(first):
                    axs[ax_nr].errorbar(x=x, y=y, fmt='x', ecolor='black', elinewidth=1, capsize=5, color = colors, label=lab)
                else:
                    axs[ax_nr].errorbar(x=x, y=y, fmt='x', ecolor='black', elinewidth=1, capsize=5, color = colors)
                axs[ax_nr].set_xticks(range(3))
                axs[ax_nr].set_xticklabels(ticklabels)
                #axs[ax_nr].plot(treatment_ind, linestyle = 'dashed')
                #axs[ax_nr].plot(control_ind, linestyle = 'dashed')
            first = False

        start_ax = len(tos) + start_ax

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].set_ylabel('Outcome Prediction')

    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('p4.png', bbox_inches='tight', dpi=1000)
    plt.clf()


def p5_plot(to_list, co_list, feat_x, all_featvals, patient_nr):

    fig, ax = plt.subplots(figsize=(15, 10))
    #ax.set_title("Partial Dependence Plots over age groups")

    patient_to = to_list[patient_nr]
    patient_co = co_list[patient_nr]

    squeetreatment= np.squeeze(to_list)
    squeecontrol= np.squeeze(co_list)

    patient_feat = feat_x[patient_nr]

    idx_list = []
    avg_pred_list_t = []
    avg_pred_list_c = []

    for tup in all_featvals:
        first_val = tup

        fv_indices = [i for i, e in enumerate(feat_x) if e == first_val]

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

    plt.plot(all_featvals,avg_pred_list_t, color='green', label='Treatment')
    plt.plot(all_featvals,avg_pred_list_c, color='blue', label='Control')
    #plt.axhline(patient_to, label="Patient Treatment Outcome ", color='green', linestyle='dashed')
    #plt.axhline(patient_co, label="Patient Control Outcome ", color='blue', linestyle='dashed')
    #plt.plot(x= patient_feat, y=patient_to, label="Patient Treatment Outcome ", color='green' ) 
    plt.errorbar(x=patient_feat, y=patient_to, fmt='o', ecolor='black', elinewidth=1, capsize=5, color = 'green', label="Patient Treatment Outcome ")
    plt.errorbar(x=patient_feat, y=patient_co, fmt='o', ecolor='black', elinewidth=1, capsize=5, color = 'blue', label="Patient Control Outcome ")
    #plt.plot(x= patient_feat, y=patient_co, label="Patient Control Outcome ", color='blue' )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel('Patient Age')
    plt.ylabel('Outcome Prediction')

    plt.savefig('p5_pat'+ str(patient_nr) + '.png', bbox_inches='tight', dpi=1000)
    plt.clf()


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

        teps_for_featvals = teps_for_featvals + [[np.mean(iteff_pred_results[0,0,8:15,patient]) for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]
        featval_sizes = [len(teps_for_featvals[x]) for x in range(len(teps_for_featvals))]
        if(f == 0):
            all_featvals = feat_group_names
        else:
            all_featvals = np.unique(feat_x)
        labs = labs + [name + str(featval) for featval in all_featvals]
    

    ordered_featvals = [np.sort([teps_for_featvals[x]]) for x in range(len(teps_for_featvals))]
    # how do i get this in one line?
    o = [x for xs in ordered_featvals for x in xs]
    #ordered_featval_list = [x for xs in o for x in xs]
    # seems like this would be ITE based, with the x axis giving indication about favouring treatment/control
    measure = [round(np.mean(vals),1) for vals in o]
    std = [round(np.std(vals),1) for vals in o]
    lower = [round(measure[x] - std[x],1) for x in range(len(measure))]
    #upper = [round(np.std(vals), 2) for vals in o]
    upper = [round(measure[x] + std[x],1) for x in range(len(measure))]

    upper_bound = round((np.max(upper) + 1), 1)
    lower_bound = round((np.min(lower) - 1), 1)
    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(effectmeasure='Avg ITE', conf_int='One Std', center= 0)
    p.colors(pointshape="x")
    ax=p.plot(figsize=(8,5), t_adjuster=0.09, max_value=upper_bound, min_value=lower_bound )
    plt.title("Individual Treatment Effects = ITE",loc="right",x=0, y=1.045)
    #plt.suptitle("Missing Data Imputation Method",x=-0.1,y=0.98)
    #ax.set_xlabel("Favours Control      Favours Chemotherapy       ", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)

    plt.savefig("p6.png", bbox_inches='tight', dpi=1000)
    plt.clf()


def p7_plot(avg_ite_pp, to_list, co_list, feature_results, feat1_nr, feat2_nr):
    avg_ite = np.mean(avg_ite_pp)
    avg_treatment_outcome = np.mean(to_list)
    avg_control_outcome = np.mean(co_list)
    # here i could exchange the mean_..._total values with the mean TEP value 
    #point_values = [mean_feature6_total, mean_treatment_outcomes, mean_control_outcomes, mean_invert_feature6_total, mean_invert_treatment_outcomes, mean_invert_control_outcomes]

    nr_categories = 4 
    ind = range(len(to_list))
    color_list = ('green','blue','blue')

    fig = plt.figure()
    gs = fig.add_gridspec(1,nr_categories, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    #fig.suptitle('Treatment and Control Outcomes for Age Groups')

    for ax in fig.get_axes():
        ax.label_outer()
    features = feature_results[:,0,0,:][0]

    mean_points_ite = np.array([avg_ite, avg_ite, avg_ite])
    mean_points_treatment= np.array([avg_treatment_outcome, avg_treatment_outcome, avg_treatment_outcome])
    mean_points_control= np.array([avg_control_outcome, avg_control_outcome, avg_control_outcome])

    feat_selection = [feat1_nr]
    labs = ['Treatment', 'Control']

    #feat_names = ['70-74', '75-79', '80-84', '85+']
    
    for feat in feat_selection:
        tos, cos, ind, x_tick_labels = get_feature_split(features, feature_results, to_list, co_list, feature_nr=feat)
        treatment_ind = np.array([ind[0], ind[0], ind[0]])
        control_ind = np.array([ind[1], ind[1], ind[1]])
        nr_featvals = len(tos)
        for ax_nr in range(nr_featvals):
            yvals = [tos[ax_nr], cos[ax_nr]]
            ticklabels = x_tick_labels[ax_nr]
            """axs[ax_nr].plot(treatment_ind, linestyle = 'dashed', label='Patient Treatment')
            axs[ax_nr].plot(control_ind, linestyle = 'dashed', label='Patient Control')"""
            axs[ax_nr].plot(mean_points_treatment, linestyle = 'dashed', color='green', label= 'Average Treatment')
            axs[ax_nr].plot(mean_points_control, linestyle = 'dashed', color='blue', label= 'Average Control')
            for x, y, colors, l in zip(range(nr_featvals), yvals, color_list, labs):
                axs[ax_nr].errorbar(x=x, y=y, fmt='x', ecolor='black', elinewidth=1, capsize=5, color = colors, label=l)
                axs[ax_nr].set_xticks(range(3))
                axs[ax_nr].set_xticklabels(ticklabels)


    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[0].set_ylabel('Outcome Prediction')
    fig.supxlabel('Age Group')

    plt.savefig('p7.png', bbox_inches='tight', dpi=1000)
    plt.clf()


def p8_plot(avg_ite, feature_results, to_list, co_list, feature_nr):

    fig, ax = plt.subplots(figsize=(10, 5))
    #ax.set_title("Partial Dependence Plots")


    feat_x = feature_results[0,0,0,:][:,feature_nr]
    all_featvals = np.unique(feat_x)

    first_feat = feat_x
    squeeite= np.squeeze(avg_ite)

    idx_list = []
    avg_pred_list = []

    for tup in all_featvals:
        first_val = tup

        fv_indices = [i for i, e in enumerate(first_feat) if e == first_val]

        patient_idx = [value for value in fv_indices]
        idx_list = idx_list + patient_idx

        #need to work out what to do with 0 values
        ite_pred_avg = np.mean([squeeite[i] for i in idx_list]) 
        
        avg_pred_list = avg_pred_list + [ite_pred_avg]
        idx_list = []

    avg_pred_list = np.reshape(avg_pred_list, (len(all_featvals)))

    plt.plot(all_featvals,avg_pred_list, color='black', label='ITE')
   
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Patient Age')
    ax.set_ylabel('Individual Treatment Effect')

    plt.savefig('p8.png', dpi=1000)
    plt.clf()


def p9_plot(avg_ite, std_ite, patient_nr):
    color_list = ['black' if x!=patient_nr else 'red' for x in range(len(avg_ite))]
    ite_with_stds = listOfTuples3(avg_ite,std_ite, color_list)
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

    #plt.title('Patient ITEs with standard deviation')
    plt.ylabel('Individual Treatment Effect')
    plt.xlabel('Individual Patients')
    plt.savefig('p9_pat' + str(patient_nr)+ '.png', dpi=1000)
    plt.clf()


def p10_plot(avg_ite, to_list, co_list):
    color_list = ['red' for x in range(len(to_list))]

    ite_with_stds = listOfTuples4(avg_ite,to_list, co_list, color_list)
    #patient_tup = ite_with_stds[patient_nr]
    sample_list = random.sample(ite_with_stds, 75)
    #sample_list.append(patient_tup)

    sorted_ite_with_stds = sorted(sample_list, key=lambda tup: tup[0])
    sorted_ites = [x[0] for x in sorted_ite_with_stds]
    sorted_treatments = [x[1] for x in sorted_ite_with_stds]
    sorted_controls= [x[2] for x in sorted_ite_with_stds]
    sorted_colors = [x[3] for x in sorted_ite_with_stds]

    sorted_treatment_effect_list = [sorted_treatments[x]-sorted_controls[x] for x in range(len(sorted_treatments))]
    ind = range(len(sorted_treatments))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    negative_ite = [x for x in sorted_treatment_effect_list if x<0]
    negative_inds = range(len(negative_ite))

    ax.bar(negative_inds, negative_ite, color='r')

    positive_ite = [x if x>0 else 0 for x in sorted_treatment_effect_list]

    ax.bar(ind,sorted_controls, color='b', label='Control Outcomes')
    #ite_bottoms = [x if x>0 else 0 for x in sorted_controls]
    ax.bar(ind,positive_ite, bottom=sorted_controls, color='r', label='Treatment Effect')

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Individual Patients")
    plt.ylabel("Outcome Prediction")
    #plt.title('Profile 10: Control Outcomes and ITE per person')

    plt.savefig('p10.png' , bbox_inches='tight', dpi=1000)
    plt.clf()


def p11_plot(all_feat1vals, all_feat2vals, feat_1, feat_2, to_list, co_list, patient_nr, nr_patients):

    nr1_buckets = len(all_feat1vals)
    nr2_buckets = len(all_feat2vals)

    # get indeces for the subgroups, to match and later extract the outcome predictions
    #teps_for_featvals = [[np.mean(results['test']['iteff_pred'][0,0,:,patient]) for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]
    teps_for_feat1vals = [np.where(feat_1 == featval)[0] for featval in all_feat1vals]
    teps_for_feat2vals = [np.where(feat_2 == featval)[0] for featval in all_feat2vals]

    indices_for_feat_combs = [[np.intersect1d(ind1,ind2) for ind2 in teps_for_feat2vals] for ind1 in teps_for_feat1vals]

    # get individual treatment and control outcomes for all feature combinations each is #different fea1vals x # different feat2val 
    treatment_outcomes_for_feat_combs = [[[to_list[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]
    control_outcomes_for_feat_combs = [[[co_list[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]

    color_list = ['red' if x!=patient_nr else 'black' for x in range(nr_patients)]
    color_list_control = ['blue' if x!=patient_nr else 'yellow' for x in range(nr_patients)]
    #colors = [[['red' if x!=patient_nr else 'black' for x in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]
    colors = [[[color_list[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]
    colors_control = [[[color_list_control[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]

    # 4. visualize
    fig = plt.figure(figsize = (10, 8))
    gs = fig.add_gridspec(nr1_buckets,nr2_buckets, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    #fig.suptitle('Outcome Predictions for different feature combinations')

    for ax in fig.get_axes():
        ax.label_outer()


    feat_names = ['70-74', '75-79', '80-84', '85+']
    y_ticks_lists = [str(featval) for featval in feat_names]
    y_ticks_lists.reverse()
    x_ticks_lists = [str(featval) for featval in all_feat2vals]

    x_ticks_list_t = ['','Treatment','']
    x_ticks_list_c = ['','Control','']
    first=True

    patient_tup = (to_list[patient_nr],co_list[patient_nr],'black','yellow')

    for ax1_nr in range(nr1_buckets):
        axs[3,ax1_nr].set_xlabel(x_ticks_lists[ax1_nr])

        for ax2_nr in range(nr2_buckets):
            #axs[ax2_nr,0].set_yticks(range(2))
            axs[ax2_nr,0].set_ylabel(y_ticks_lists[ax2_nr])
            #axs[0,ax1_nr].set_xlabel(x_ticks_lists[ax1_nr])

            
            ite_with_stds = listOfTuples4(treatment_outcomes_for_feat_combs[ax1_nr][ax2_nr], control_outcomes_for_feat_combs[ax1_nr][ax2_nr], colors[ax1_nr][ax2_nr], colors_control[ax1_nr][ax2_nr])
            

            #sample_list = random.sample(ite_with_stds, 100)
            #sample_list.append(patient_tup)
            #ax_len = len(teps_for_featvals[ax_nr])

            if(len(ite_with_stds) < 50):
                sample_list = ite_with_stds
            else:
                sample_list = random.sample(ite_with_stds, 50)

            if(all_feat1vals[ax1_nr] == feat_1[patient_nr] and all_feat2vals[ax2_nr] == feat_2[patient_nr] and (patient_tup not in sample_list)):
                sample_list.append(patient_tup)

            sorted_ite_with_stds = sorted(sample_list, key=lambda tup: tup[0]-tup[1])
            sorted_treatments = [x[0] for x in sorted_ite_with_stds]
            sorted_controls= [x[1] for x in sorted_ite_with_stds]
            sorted_colors = [x[2] for x in sorted_ite_with_stds]
            sorted_color_control = [x[3] for x in sorted_ite_with_stds]

            sorted_treatment_effect_list = [sorted_treatments[x]-sorted_controls[x] for x in range(len(sorted_treatments))]
            ind = range(len(sorted_treatments))

            for x, sc, ite, colo, ccolor in zip(ind, sorted_controls, sorted_treatment_effect_list, sorted_colors, sorted_color_control):
                if(ite > 0):
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

    #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.supylabel('Age Group')
    fig.supxlabel('Tumor Grade')
    plt.savefig('p11_pat' + str(patient_nr)+'.png', dpi=1000, bbox_inches= 'tight')
    plt.clf()


def p12_plot(avg_ite):
    fig = plt.figure()
    #plt.suptitle('Boxplot for expected outcomes over test population')

    avg = list(list(avg_ite)[0])

    for ax in fig.get_axes():
        ax.label_outer()

    plt.boxplot(avg)
    #plt.xlabel('Individual Treatment Effect Predictions')
    plt.ylabel('Individual Treatment Effect')

    plt.savefig('p12.png', dpi=1000)
    plt.clf()


def p13_plot(avg_ite, to_list, co_list,feat_results, iteff_pred_results, feat1_nr, feat2_nr, patient_nr):
    ite_with_stds = listOfTuples3(avg_ite,to_list, co_list)
    sorted_ite_with_stds = sorted(ite_with_stds, key=lambda tup: tup[0])
    sorted_ites = [x[0] for x in sorted_ite_with_stds]
    sorted_treatments = [x[1] for x in sorted_ite_with_stds]
    sorted_controls= [x[2] for x in sorted_ite_with_stds]

    fig, ax = plt.subplots()
    #ax.set_title("Two Way Partial Dependence Plot over ITEs")

    big_X = np.squeeze(np.mean(feat_results[:,0,:,:], axis=(1)))

    true_Y = np.squeeze(np.mean(iteff_pred_results[:,0,:,:], axis=(1)))

    feat_1 = feat_results[0,0,0,:][:,feat1_nr]
    feat_2 = feat_results[0,0,0,:][:,feat2_nr]

    patient_f1val = feat_1[patient_nr]
    patient_f2val = feat_2[patient_nr]
    # needs to be squeezed, otherwise this is interpreted as multi-output regressor
    squeeite= np.squeeze(avg_ite)

    dum = DummyRegressor(wanted_output=squeeite)
    dum.fit(big_X,true_Y)

    pdp_plot = PartialDependenceDisplay.from_estimator(
        estimator=dum,
        X=big_X,
        features=[(feat1_nr,feat2_nr)], # the features to plot
        tep_results=iteff_pred_results[:,0,:,:],
        method='brute',
        ax=ax,
    )


    if(patient_f1val == 70):
        patient_f1val = patient_f1val+2
    plt.axvline(patient_f1val, label='Patient Features', color='black')

    plt.axhline(patient_f2val, color='black')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Patient Age')
    plt.ylabel('Tumor Grade')
    plt.savefig('p13_pat' + str(patient_nr) + '.png', bbox_inches='tight', dpi=1000)
    plt.clf()


def p14_plot(avg_ite, std_ite, to_list, co_list, feature_results, feat1_nr, feat2_nr):
    #avg_ite = np.mean(results['test']['iteff_pred'][:,0,:,:], axis=(1))

    avg_ite_per_person = list(list(avg_ite)[0])
    std_ite_per_person = list(list(std_ite)[0])

    prc_to_sample = 0.06
    
    # 1. calculate how many patients have each feature expression
    feat_1 = feature_results[0,0,0,:][:,feat1_nr]
    feat_2 = feature_results[0,0,0,:][:,feat2_nr]
    all_feat1vals = np.unique(feat_1)
    all_feat2vals = np.unique(feat_2)
    nr1_buckets = len(all_feat1vals)
    nr2_buckets = len(all_feat2vals)

    # get indeces for the subgroups, to match and later extract the outcome predictions
    #teps_for_featvals = [[np.mean(results['test']['iteff_pred'][0,0,:,patient]) for patient in np.where(feat_x == featval)[0]] for featval in all_featvals]
    teps_for_feat1vals = [np.where(feat_1 == featval)[0] for featval in all_feat1vals]
    teps_for_feat2vals = [np.where(feat_2 == featval)[0] for featval in all_feat2vals]

    indices_for_feat_combs = [[np.intersect1d(ind1,ind2) for ind2 in teps_for_feat2vals] for ind1 in teps_for_feat1vals]

    # get individual treatment and control outcomes for all feature combinations each is #different fea1vals x # different feat2val 
    treatment_outcomes_for_feat_combs = [[[to_list[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]
    control_outcomes_for_feat_combs = [[[co_list[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]
    colors = [[['red' for x in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]

    ite_preds_for_feat_combs = [[[avg_ite_per_person[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]
    std_preds_for_feat_combs = [[[std_ite_per_person[indices] for indices in layer2] for layer2 in layer1] for layer1 in indices_for_feat_combs]

    # 4. visualize
    fig = plt.figure(figsize=(15,10))
    gs = fig.add_gridspec(nr1_buckets,nr2_buckets, hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    #fig.suptitle('Individual Treatment Effect values for patients in subpopulations')

    for ax in fig.get_axes():
        ax.label_outer()
    
    feat_names = ['70-74', '75-79', '80-84', '85+']
    y_ticks_lists = [str(featval) for featval in feat_names]
    y_ticks_lists.reverse()
    x_ticks_lists = [str(featval) for featval in all_feat2vals]
    #y_ticks_lists.reverse()

    x_ticks_list_t = ['','Treatment','']
    x_ticks_list_c = ['','Control','']

    #sample_size = np.sum(featval_sizes)

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

            #ind = range(len(sorted_colors))

            for x, y, std in zip(range(len(sorted_treatment_effect_list)), sorted_treatment_effect_list, sorted_stds):
                axs[ax1_nr,ax2_nr].errorbar(x=x, y=y, yerr=std, fmt='o', ecolor='black', elinewidth=1, capsize=5, color = 'black')

    #axs.set_ylabel('Tumor Grade')
    fig.supylabel('Age Group')
    fig.supxlabel('Tumor Grade')
    #axs.set_xlabel('Age Group')
    plt.savefig('p14.png', dpi=1000)
    plt.clf()


def p15_plot(avg_ite, teps_for_featval, featval_labs, patient_nr):
    ordered_featvals = [np.sort([teps_for_featval[x]]) for x in range(len(teps_for_featval))]
    o = [x for xs in ordered_featvals for x in xs]

    feat_names = ['Age 70-74', 'Age 75-79', 'Age 80-84', 'Age 85+']

    # seems like this would be ITE based, with the x axis giving indication about favouring treatment/control
    patient_ite = avg_ite[patient_nr]
    measure = [round(np.mean(vals),1) for vals in o]
    std = [round(np.std(vals),1) for vals in o]
    lower = [round(measure[x] - std[x],1) for x in range(len(measure))]
    #upper = [round(np.std(vals), 2) for vals in o]
    upper = [round(measure[x] + std[x],1) for x in range(len(measure))]

    upper_bound = round((np.max(upper) + 1), 1)
    lower_bound = round((np.min(lower) - 1), 1)
    p = EffectMeasurePlot(label=feat_names, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(effectmeasure='Avg ITE', conf_int='One Std', center=0)
    p.colors(pointshape="x")
    
    ax=p.plot(figsize=(8,5), t_adjuster=0.09, max_value=upper_bound, min_value=lower_bound )
    ax.axvline(patient_ite, color='red', label='Patient ITE')
    #plt.title("Individual Treatment Effects for different Age Groups",loc="right",x=0, y=1.045)
    #plt.suptitle("Missing Data Imputation Method",x=-0.1,y=0.98)
    #ax.set_xlabel("Favours Control      Favours Chemotherapy       ", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    ax.legend()

    plt.savefig("p15_pat" + str(patient_nr) + ".png", bbox_inches='tight', dpi=1000)
    plt.clf()


def p16_plot(avg_ite, std_ite):
    ite_with_stds = listOfTuples2(avg_ite, std_ite)
    sorted_ite_with_stds = sorted(ite_with_stds, key=lambda tup: tup[0])
    sorted_ites = [x[0] for x in sorted_ite_with_stds]
    sorted_stds = [x[1] for x in sorted_ite_with_stds]

    # sortiere ITEs in einerbuckets
    fig = plt.figure()
    max_ite = int(sorted_ites[len(sorted_ites)-1]) + 2  # +1 somehow does not create the last bucket at the top
    min_ite = int(sorted_ites[0]) - 1
    bins = np.array(range(min_ite,max_ite))

    ites_in_bins = pd.cut(sorted_ites,bins)
    ite_hist = np.histogram(sorted_ites,bins)

    plt.hist(sorted_ites,list(ite_hist[1]), color= 'black')
    plt.xlabel('Individual Treatment Effect')
    plt.ylabel('Number of Patients')
    #plt.title('ITE Histogram over test population')
    plt.savefig('p16.png', dpi=1000)
    plt.clf()


def p17_plot(to_list, co_list, feature_results, feat1_nr, feat2_nr):
    fig, ax = plt.subplots(figsize=(15, 10))

    #ax.set_title("Partial Dependence Plots")


    squeetreatment= np.squeeze(to_list)
    squeecontrol= np.squeeze(co_list)
    feat_list = [feat1_nr,feat2_nr]
    names_list = ['Age', 'Tumor Grade']
    colors = [['green', 'blue'], ['yellow', 'red']]

    for feat_nr, feat_name, col in zip(feat_list,names_list, colors):
        feat_x = feature_results[0,0,0,:][:,feat_nr]
        all_featvals = np.unique(feat_x)

        first_feat = feat_x
        # fv_indices are the index values for any given feature value
        idx_list = []

        avg_pred_list_t = []
        avg_pred_list_c = []
        for tup in all_featvals:
            first_val = tup
            #second_val = tup[1]

            fv_indices = [i for i, e in enumerate(first_feat) if e == first_val]
            #sv_indices = [i for i, e in enumerate(second_feat) if e == second_val]

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


def p18_plot(avg_ite, t_outcomes, c_outcomes, patient_nr):
    chosen_patient_treatment_outcome = t_outcomes[patient_nr]
    chosen_patient_control_outcome = c_outcomes[patient_nr]

    value_tuple = listOfTuples3(avg_ite, t_outcomes, c_outcomes)
    sorted_value_tuple = sorted(value_tuple, key=lambda tup: tup[0])
    sorted_ites = [x[0] for x in sorted_value_tuple]
    sorted_treatments = [x[1] for x in sorted_value_tuple]
    sorted_controls= [x[2] for x in sorted_value_tuple]

    min_treatment = int(sorted_treatments[len(sorted_treatments)-1]) + 2  # +1 somehow does not create the last bucket at the top
    max_treatment = int(sorted_treatments[0]) - 1
    min_control = int(sorted_controls[len(sorted_controls)-1]) + 2  # +1 somehow does not create the last bucket at the top
    max_control = int(sorted_controls[0]) - 1

    # need bins for min control and max treatment supposedly
    bin_nr = max_treatment - min_control

    xxx = list(zip(t_outcomes, c_outcomes))

    a = np.array(xxx)
    plt.hist(a,bin_nr, label=['Treatment', 'Control'], color=['green', 'blue']) # density=True, histtype='bar',

    plt.axvline(chosen_patient_treatment_outcome, label="Patient predicted Treatment Outcome", color='orange')
    plt.axvline(chosen_patient_control_outcome, label="Patient predicted Control Outcome", color='yellow')

    plt.xlabel('Outcome Prediction')
    plt.ylabel('Number of Patients')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('p18_pat' + str(patient_nr) +'.png', bbox_inches='tight', dpi=1000)
    plt.clf()

