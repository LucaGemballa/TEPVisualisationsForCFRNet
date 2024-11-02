import csv
import time
import pandas as pd
import math
import numpy as np

"""
The data selection from SEER that is refined here is inspired by the paper:
"""

def changesToColumns():
    # add your file path here
    orig = pd.read_csv(r'C:\Users\lucag\Documents\VSCodeProjekte\TEPVisualisationsForCFRNet\data\unrefined_study_data.csv')

    # this feature turned out to contain no actual information
    orig = orig.drop(['Lymph Node Size Recode (2010+)'], axis=1)


    # this line is also for the 4 metastasis variants
    orig = orig.replace({'No': 0, 'Yes': 1})
    orig = orig.replace({'No/Unknown': 0})
    
    # Age is defined in ranges of 5 years. I always choose the lowest bound
    age_statements = ['70-74 years', '75-79 years', '80-84 years', '85+ years']
    target_ages = [70,75,80,85]
    for age, target in zip(age_statements,target_ages):
        pos = orig["Age recode with <1 year olds"] == age
        orig.loc[pos,"Age recode with <1 year olds"] = target

    # drop every row containing carcinoma types except for ductal, lobular and ductal-lobular
    pos1 = orig["ICD-O-3 Hist/behav"] == '8500/3: Infiltrating duct carcinoma, NOS' 
    pos2 = orig["ICD-O-3 Hist/behav"] == '8520/3: Lobular carcinoma, NOS' 
    pos3 = orig["ICD-O-3 Hist/behav"] ==  '8522/3: Infiltrating duct and lobular carcinoma'
    pos = pos1 | pos2 | pos3
    orig = orig.drop(orig[~pos].index)


    # for grade recode always give the target grade without text
    # drop unknowns
    pos1 = orig["Grade Recode (thru 2017)"] == 'Unknown'
    orig = orig.drop(orig[pos1].index)
    grade_statements = ['Well differentiated; Grade I', 'Moderately differentiated; Grade II', 'Poorly differentiated; Grade III', 'Undifferentiated; anaplastic; Grade IV']
    target_grades = [1,2,3,4]
    for grade, target in zip(grade_statements,target_grades):
        pos = orig["Grade Recode (thru 2017)"] == grade
        orig.loc[pos,"Grade Recode (thru 2017)"] = target


    # for T(umor) N(odes) M(etastasis) values drop blank rows
    pos1 = orig["Derived AJCC T, 7th ed (2010-2015)"] == 'Blank(s)'
    orig = orig.drop(orig[pos1].index)

    # include only patients who had surgery before chemotherapy/ those who only had surgery
    pos1 = orig['RX Summ--Systemic/Sur Seq (2007+)'] == 'Systemic therapy after surgery'
    orig = orig.drop(orig[~pos1].index)

    # drop missing values for lymph nodes 
    pos1 = orig['LN Positive Axillary Level I-II Recode (2010+)'] == 'Not documented; Not assessed or unknown if assessed'
    pos2 = orig['LN Positive Axillary Level I-II Recode (2010+)'] == 'Positive aspiration or needle core biopsy of lymph node(s)'
    pos3 = orig['LN Positive Axillary Level I-II Recode (2010+)'] == 'Positive nodes, number unspecified'
    pos = pos1 | pos2 | pos3
    orig = orig.drop(orig[pos].index)

    # remove filler zeros from the values
    orig['LN Positive Axillary Level I-II Recode (2010+)'] = orig['LN Positive Axillary Level I-II Recode (2010+)'].str.lstrip('0')

    # where no metastasis were found in lymph nodes fill 0 as a number instead
    pos = orig['LN Positive Axillary Level I-II Recode (2010+)'] == 'All ipsilateral axillary nodes examined negative'
    orig.loc[pos,'LN Positive Axillary Level I-II Recode (2010+)'] = 0
    orig['LN Positive Axillary Level I-II Recode (2010+)'] = orig['LN Positive Axillary Level I-II Recode (2010+)'].astype(int)

    pos1 = orig['Tumor Size Over Time Recode (1988+)'] == 'Unknown or size unreasonable (includes any tumor sizes 401-989)'
    pos2 = orig['Tumor Size Over Time Recode (1988+)'] == '990 (microscopic focus)'
    pos3 = orig['Tumor Size Over Time Recode (1988+)'] == '998 (site-specific code)'
    pos4 = orig['Tumor Size Over Time Recode (1988+)'] == '000 (no evidence of primary tumor)'
    pos = pos1 | pos2 | pos3 | pos4
    orig = orig.drop(orig[pos].index)
    orig['Tumor Size Over Time Recode (1988+)'] = orig['Tumor Size Over Time Recode (1988+)'].str.lstrip('0')
    orig['Tumor Size Over Time Recode (1988+)'] = orig['Tumor Size Over Time Recode (1988+)'].astype(int)

    orig['CS tumor size (2004-2015)'] = orig['CS tumor size (2004-2015)'].str.lstrip('0') 
    orig['CS tumor size (2004-2015)'] = orig['CS tumor size (2004-2015)'].astype(int)


    # replace all 000 before stripping of leading 0s
    pos = orig["CS lymph nodes (2004-2015)"] == '000'
    orig.loc[pos,"CS lymph nodes (2004-2015)"] = '01'
    orig['CS lymph nodes (2004-2015)'] = orig['CS lymph nodes (2004-2015)'].str.lstrip('0') 
    pos = orig["CS lymph nodes (2004-2015)"] == '01'
    orig.loc[pos,"CS lymph nodes (2004-2015)"] = 0
    orig['CS lymph nodes (2004-2015)'] = orig['CS lymph nodes (2004-2015)'].astype(int)
    
    
    pos1 = orig['SEER Combined Mets at DX-bone (2010+)'] == 'Unknown'
    pos2 = orig['SEER Combined Mets at DX-brain (2010+)'] == 'Unknown'
    pos3 = orig['SEER Combined Mets at DX-liver (2010+)'] == 'Unknown'
    pos4 = orig['SEER Combined Mets at DX-lung (2010+)'] == 'Unknown'
    pos = pos1 | pos2 | pos3 | pos4
    orig = orig.drop(orig[pos].index)
    
    pos1 = orig['Median household income inflation adj to 2022'] == 'Unknown/missing/no match/Not 1990-2022'
    orig = orig.drop(orig[pos1].index)
    
    income_statements = orig['Median household income inflation adj to 2022'].unique()
    target_incomes = [9,9.5,10,11,12,8.5,8,7.5,7,6.5,6,5.5,5,4.5,4,3.5]
    for income, target in zip(income_statements,target_incomes):
        pos = orig["Median household income inflation adj to 2022"] == income
        orig.loc[pos,"Median household income inflation adj to 2022"] = target

    orig['CS Mets Eval (2004-2015)'] = orig['CS Mets Eval (2004-2015)'].astype(int)
    
    # get_dummies builds a One-Hot encoding for the given columns
    orig = pd.get_dummies(data = orig, columns= ['Race recode (W, B, AI, API)', 'ICD-O-3 Hist/behav', 'Marital status at diagnosis', 'Derived AJCC T, 7th ed (2010-2015)',
     'Derived AJCC N, 7th ed (2010-2015)', 'Derived AJCC M, 7th ed (2010-2015)', 'RX Summ--Systemic/Sur Seq (2007+)'])

    # get outcome and treatment, then remove their columns from orig
    outcome = orig['Survival months']
    treatment = orig['Chemotherapy recode (yes, no/unk)']

    orig.to_csv(r'C:\Users\lucag\Documents\VSCodeProjekte\TEPVisualisationsForCFRNet\data\refined_study_data.csv', index=False)
    orig = orig.drop(['Survival months', 'Chemotherapy recode (yes, no/unk)'], axis=1)
    
    orig.to_csv(r'C:\Users\lucag\Documents\VSCodeProjekte\TEPVisualisationsForCFRNet\data\refined_study_data_complete.csv', index=False)

    # split orig into train and test
    split_dist = math.floor(orig.shape[0] * 0.8)

    print('Training data contains ', split_dist ,'patients')
    origtrain = orig.iloc[:split_dist, :]
    ttrain = treatment.iloc[:split_dist]
    outtrain = outcome.iloc[:split_dist]

    origtest = orig.iloc[split_dist:, :]
    ttest = treatment.iloc[split_dist:]
    outtest = outcome.iloc[split_dist:]

    dftrainx = origtrain.to_numpy()
    dftraint = ttrain.to_numpy()
    dftrainout = outtrain.to_numpy()

    # columns 12 and 13 are outcome and treatment
    dftestx = origtest.to_numpy()
    dftestt = ttest.to_numpy()
    dftestout = outtest.to_numpy()
    print(dftestx.shape, dftestt.shape,dftestout.shape)

    # with npz can save multiple arrays at once: split features, treatment and outcome
    np.savez('masterstudytrain', dftrainx, dftraint, dftrainout)
    np.savez('masterstudytest', dftestx, dftestt, dftestout)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    changesToColumns()


