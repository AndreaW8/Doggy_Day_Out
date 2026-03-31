#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:37:20 2025

@author: dragon


Williamson County Regional Animal Shelter, in Georgetown, TX, runs a one day 
foster program called Doggy Day Out. This program allows people over 18 to 
choose a dog to take on an outing for the day. The shelter will sign the person 
up as a foster for that dog and only for that day. Then they take the dog on 
the outing. They can: Go to a park. Get a pup cup. Take a nap on the couch. 
And most importantly, HAVE FUN!!!¶

We will take data from Williamson County Regional Animal Shelter to assess the 
postive impact of this program. Are Doggy Day Out dogs apoted sooner? Does it 
help dogs who have been in shelter longer get adopted?


File input: Output___per_Animal_ID_n_ddo_cnt_NO_PUPPIES_df.csv


This code performs a comprehensive machine learning workflow to predict whether a 
dog's operation outcome is adoption or not, using features from a dataset 
filtered to exclude puppies and data before March 2023. Key steps include:

    --Data cleaning and preprocessing with one-hot encoding and scaling.
    
    --Creating a binary target variable (Adoption vs. others).

    
    --Feature selection using mutual information (MI) to pick the top 20 features.
    
    --Dimensionality reduction with PCA and visualization of explained variance.
    
    --Hyperparameter tuning and evaluation of Random Forest and SVM classifiers 
    across various dataset versions (all features, all features w/o DDO, 
                                     selected MI features, selected MI featuresw/o DDO, 
                                     PCA data, oversampled, and undersampled).
    
    --Comparing models based on cross-validation and test metrics (accuracy and multiple F1 scores).
    
    --Extracting and visualizing feature importances from the best-performing models.


"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# from pandas.testing import assert_frame_equal
from collections import Counter
import random
# import os


# random seed
project_seed = 42
random.seed(project_seed)



# -----------------------------------------------------------------------------
# Functions!!!! ***
# -----------------------------------------------------------------------------
def rf_search(X_, X_train_, X_test_, y_train_, y_test_, seed):
    """
    
    performs randomized hyperparameter tuning for a Random Forest classifier using cross-validation, 
    reports model performance on the test set, and returns DataFrames summarizing the search results 
    and feature importances.
    
    Returns: 
    rf_search_df: A DataFrame summarizing the hyperparameter search results and corresponding cross-validation metrics.

    rf_scores_df: A DataFrame detailing each feature’s importance in the final model, 
                    sorted by their contribution to prediction accuracy.
    
    """
    # parameter grid for Random Forest (Ensemble)
    rf_params = {'n_estimators':[10,25,50,100,200,300], 
                 'criterion':['gini','entropy','log_loss'],
                 'max_depth':[None,4, 6, 8, 10, 20, 50, 100],
                 'min_samples_split':[2,5,10,20], 
                 'max_features':[None,'sqrt','log2'],
                 'max_samples':[None,0.10,0.50,0.75],
                 'class_weight': ['balanced', 'balanced_subsample', None],
                 'random_state':[seed]
                }
    
    scoring_metric_I_care_about = 'f1_macro'
    # https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
    # https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/
    # 'weighted' - favouring the majority class
    # 'micro' - no favouring any class in particular.
    # 'macro'  -  bigger penalisation when your model does not perform well with the minority classes.


    rf_search = RandomizedSearchCV(RFC(),rf_params, # positional arguments, model and parameter grid
                                          n_iter=300,
                                          scoring=['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted'],
                                          refit = scoring_metric_I_care_about,
                                          # see scoring options here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                          cv=10,
                                          random_state=project_seed,
                                          # n_jobs=1)
                                          n_jobs=-1) 
    
    
    # rf_search = GridSearchCV(RFC(),rf_params, # positional arguments, model and parameter grid
    #                                       scoring=['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted'],
    #                                       refit = scoring_metric_I_care_about,
    #                                       # see scoring options here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #                                       cv=10,
    #                                       # n_jobs=1)
    #                                       n_jobs=-1) 


    rf_search.fit(X_train_,y_train_)

    rf_search_data = {'params': rf_search.cv_results_['params'], 
                      'mean_test_accuracy': rf_search.cv_results_['mean_test_accuracy'], 
                      'mean_test_f1_macro': rf_search.cv_results_['mean_test_f1_macro'],
                      'mean_test_f1_micro': rf_search.cv_results_['mean_test_f1_micro'],
                      'mean_test_f1_weighted': rf_search.cv_results_['mean_test_f1_weighted']
                      }
    rf_search_df = pd.DataFrame(rf_search_data)
    rf_search_df = rf_search_df.sort_values(by='mean_test_'+scoring_metric_I_care_about, ascending=False) 
    rf_search_df = rf_search_df.reset_index(drop=True)


    # print('y.value_counts():')
    # print(y.value_counts())
    # print("------------------------------------------------------")

    # # Get the best Random Forest model
    best_rf_model = rf_search.best_estimator_
    
    # y_pred = rf_search.predict(X_test_)

    # print("------------------------------------------------------")
    # # print("ALL THE Features used! ")
    # print("Classification Report with hold-out sample for best fit of this model:...\n")
    # print(type(RFC()))
    # print(classification_report(y_test_,y_pred))
    # print("------------------------------------------------------")
    # print(f'Best Hyperparameters: {rf_search.best_params_}')
    # print("------------------------------------------------------")
    feature_importances = best_rf_model.feature_importances_


    rf_scores_df = pd.DataFrame(zip(X_.columns, feature_importances), columns=['Feature', 'feature_importances'])
    rf_scores_df = rf_scores_df.sort_values(by='feature_importances', ascending=False) # Sort by 'feature_importances' 
    rf_scores_df = rf_scores_df.reset_index(drop=True)
    
    
    return rf_search_df, rf_scores_df



def SVM_generic_search(X_train_, y_train_, SVM_params):
    
    """
    automates hyperparameter tuning and evaluation for a Support Vector Machine (SVM) classifier using randomized search. 
    
    Returns: 
    SVM_search_df: A DataFrame summarizing the hyperparameter search results and corresponding cross-validation metrics.
    """
    
    # SVM classifier
    svm = SVC()
    
    
    scoring_metric_I_care_about = 'f1_macro'
    # https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
    # https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/
    # 'weighted' - favouring the majority class
    # 'micro' - no favouring any class in particular.
    # 'macro'  -  bigger penalisation when your model does not perform well with the minority classes.

    # # Create a SearchCV object  
    # SVM_search = RandomizedSearchCV(svm, SVM_params,
    #                                    n_iter=100,
    #                                   scoring=['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted'],
    #                                   refit = scoring_metric_I_care_about,
    #                                   # see scoring options here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #                                   cv=10,
    #                                   random_state=project_seed,
    #                                   # verbose=3,
    #                                    # n_jobs=1)
    #                                    n_jobs=-1) 
    
    # Create a SearchCV object  
    SVM_search = GridSearchCV(svm, SVM_params,
                                      scoring=['accuracy', 'f1_macro', 'f1_micro', 'f1_weighted'],
                                      refit = scoring_metric_I_care_about,
                                      # see scoring options here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                      cv=10,
                                      # verbose=3,
                                       # n_jobs=1)
                                       n_jobs=-1) 

    # Perform the grid search
    SVM_search.fit(X_train_, y_train_)



    SVM_search_data = {'params': SVM_search.cv_results_['params'], 
                      'mean_test_accuracy': SVM_search.cv_results_['mean_test_accuracy'], 
                      'mean_test_f1_macro': SVM_search.cv_results_['mean_test_f1_macro'],
                      'mean_test_f1_micro': SVM_search.cv_results_['mean_test_f1_micro'],
                      'mean_test_f1_weighted': SVM_search.cv_results_['mean_test_f1_weighted']
                      }
    SVM_search_df = pd.DataFrame(SVM_search_data)
    SVM_search_df = SVM_search_df.sort_values(by='mean_test_'+scoring_metric_I_care_about, ascending=False) 
    SVM_search_df = SVM_search_df.reset_index(drop=True)
    
    
    return SVM_search_df



def svm_search(X_train_, X_test_, y_train_, y_test_):
    """
    performs comprehensive hyperparameter tuning for both linear and RBF-kernel 
    Support Vector Machine (SVM) classifiers using randomized search with cross-validation. 
    
    Returns: 
    SVM_search_df: A DataFrame summarizing the hyperparameter search results from both linear and RBF-kernel searches
    and corresponding cross-validation metrics.
    """
    SVM_linear_params = {
        'C': [0.1, 1, 10, 500, 1000],
                # C: Regularization parameter. The strength of the regularization 
                    # is inversely proportional to C.
                    # Must be strictly positive. The penalty is a squared l2 penalty. 
        'kernel': ['linear'],    
        'random_state':[project_seed],
        'class_weight': ['balanced', None],
        'max_iter':[500000]
    }


    SVM_rbf_params = {
        'C': [0.1, 1, 10, 500, 1000],
                # C: Regularization parameter. The strength of the regularization is inversely proportional to C.
                    # Must be strictly positive. The penalty is a squared l2 penalty. 
        'gamma': ['scale', 'auto', 0.1, 50, 100],  
                # 'gamma': Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
        'kernel': ['rbf'],
        'random_state':[project_seed],
        'class_weight': ['balanced', None],
        'max_iter':[500000]
    }

    SVM_linear_search_df = SVM_generic_search(X_train_, y_train_, SVM_linear_params)

    SVM_rbf_search_df = SVM_generic_search(X_train_, y_train_, SVM_rbf_params)

    SVM_search_df = pd.concat([SVM_linear_search_df, SVM_rbf_search_df])
    
    
    scoring_metric_I_care_about = 'f1_macro'
    # https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
    # https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/
    # 'weighted' - favouring the majority class
    # 'micro' - no favouring any class in particular.
    # 'macro'  -  bigger penalisation when your model does not perform well with the minority classes.
    
    
    
    SVM_search_df = SVM_search_df.sort_values(by='mean_test_'+scoring_metric_I_care_about, ascending=False) 
    SVM_search_df = SVM_search_df.reset_index(drop=True)


    # svm = SVC()

    # best_model_prams = SVM_search_df['params'].iloc[0]

    # svm_model = SVC(**best_model_prams)
    # svm_model.fit(X_train_, y_train_)
    # y_pred = svm_model.predict(X_test_)

    # print("------------------------------------------------------")
    # print("Classification Report with hold-out sample for best fit of this model:...\n")
    # print(type(svm))
    # print(classification_report(y_test_,y_pred))
    # print("------------------------------------------------------")
    # print(f'Best Hyperparameters: {best_model_prams}')
    # print("------------------------------------------------------")
    
    # # SVM_feature_importances = best_SVM_model.coef_
    # # AttributeError: coef_ is only available when using a linear kernel
    
    return SVM_search_df



def print_classification_report(model, search_df, descriptive_text, X_train_, X_test_, y_train_, y_test_):
    """
    Takes best hyperparameters from a search results DataFrame. 
    Initializes a model, trains on train data, and predicts on testing data.
    Prints out classification report on test data
    
    """

    best_model_prams = search_df['params'].iloc[0]
    
    if model == "SVM":
        model = SVC(**best_model_prams)
    else:
        model = RFC(**best_model_prams)
        
    model.fit(X_train_, y_train_)
    y_pred = model.predict(X_test_)

    print()
    print("------------------------------------------------------")
    print(descriptive_text)
    print("------------------------------------------------------")
    print('CV scores from best model:')
    pd.set_option('display.max_columns', None)
    print(search_df.iloc[[0], -4:])
    pd.reset_option('display.max_columns')
    print("------------------------------------------------------")
    print("Classification Report with hold-out sample for best fit of this model:...")
    print("------------------------------------------------------")
    print(type(model))
    print(classification_report(y_test_,y_pred))
    print("------------------------------------------------------")
    print(f'Best Hyperparameters: {best_model_prams}')
    print("------------------------------------------------------\n\n")
    
    
def top_models_based_on_test_data(model, search_df, search_df_name, X_train_, X_test_, y_train_, y_test_):
    """
    Takes best hyperparameters from a search results DataFrame. 
    Initializes a model, trains on train data, and predicts on testing data.
    
    Returns: 
        Dataframe of model result metrics from testing data.
    """

    best_model_prams = search_df['params'].iloc[0]

    if model == "SVM":
        model = SVC(**best_model_prams)
    else:
        model = RFC(**best_model_prams)
        
    model.fit(X_train_, y_train_)
    y_pred = model.predict(X_test_)
        
        
    metric_dict = {
            'best_prams': [best_model_prams],
            'accuracy': [accuracy_score(y_test_, y_pred)],
            'f1_macro': [f1_score(y_test_, y_pred, average='macro')],
            'f1_micro': [f1_score(y_test_, y_pred, average='micro')],
            'f1_weighted': [f1_score(y_test_, y_pred, average='weighted')],
            'run': [search_df_name]
        }
    
    top_model_df = pd.DataFrame(metric_dict)
    
    # scoring_metric_I_care_about = 'f1_macro'
    # # https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
    # # https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/
    # # 'weighted' - favouring the majority class
    # # 'micro' - no favouring any class in particular.
    # # 'macro'  -  bigger penalisation when your model does not perform well with the minority classes.
    
    return top_model_df
        
        
            

# -----------------------------------------------------------------------------
# Read in DATA!!!! ***
# -----------------------------------------------------------------------------
# per_stay_n_ddo_cnt_df = pd.read_csv('Data/Output__Animal_ID_per_stay_n_ddo_cnt_df.csv')
# per_stay__no_puppies__ddo_cnt_df = pd.read_csv('Data/Output__Animal_ID_per_stay_n_ddo_cnt_NO_PUPPIES_df.csv')
per_dog__no_puppies__ddo_cnt_df = pd.read_csv('Data/Output___per_Animal_ID_n_ddo_cnt_NO_PUPPIES_df.csv')



# --------------------------------------------------
# remove data point before DDO program in 3/2023
# --------------------------------------------------
per_dog__no_puppies__ddo_cnt_df['Date/Time'] = pd.to_datetime(per_dog__no_puppies__ddo_cnt_df['Date/Time'])

DDO_start_date = pd.to_datetime('2023-03-01')
per_dog__no_puppies__ddo_cnt_df = per_dog__no_puppies__ddo_cnt_df[per_dog__no_puppies__ddo_cnt_df['Date/Time'] >= DDO_start_date]



# --------------------------------------------------
# Ensure no puppies
# -------------------
per_dog__no_puppies__ddo_cnt_df = per_dog__no_puppies__ddo_cnt_df[per_dog__no_puppies__ddo_cnt_df['Age_Group'] != 'Baby']



# --------------------------------------------------
# drop cols that aren't needed
# --------------------------------------------------
per_dog__no_puppies__ddo_cnt_df = per_dog__no_puppies__ddo_cnt_df.drop(['Animal_ID', 'Date/Time'], axis=1)  # drop cols that aren't needed

per_dog__no_puppies__ddo_cnt_df['Size_shifted'].value_counts()



# --------------------------------------------------
# Target col distribution
# --------------------------------------------------

counts = per_dog__no_puppies__ddo_cnt_df['Operation_Type'].value_counts()
percentages = per_dog__no_puppies__ddo_cnt_df['Operation_Type'].value_counts(normalize=True) * 100


og_target_summary_df = pd.DataFrame({
    'Count': counts,
    'Percentage': percentages.round(2)  
})

# print(og_target_summary_df)
#                           Count  Percentage
# Operation_Type                             
# Adoption                   2048       56.00
# Return to Owner/Guardian    966       26.42
# Transfer Out                565       15.45
# Euthanasia                   53        1.45
# Died                         20        0.55
# Missing                       3        0.08
# Clinic Out                    1        0.03
# Admin Missing                 1        0.03

# Plot the bar chart
plt.figure(figsize=(8, 5))
counts.plot(kind='bar', color='skyblue')
plt.title('Counts of Values in Operation_Type')
plt.xlabel('Operation_Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# --------------------------------------------------
# filter out class ***
# --------------------------------------------------

# removed 'Return to Owner/Guardian' becuse this woudl add noise to my data.
# Return to Owner dog will never be adopted
per_dog__no_puppies__ddo_cnt_df = per_dog__no_puppies__ddo_cnt_df[
    ~per_dog__no_puppies__ddo_cnt_df['Operation_Type'].isin(['Return to Owner/Guardian'])
]

# per_dog__no_puppies__ddo_cnt_df = per_dog__no_puppies__ddo_cnt_df[
#     ~per_dog__no_puppies__ddo_cnt_df['Operation_Type'].isin(['Missing', 'Clinic Out', 'Admin Missing', 'Euthanasia', 'Died'])
# ]


# print(per_dog__no_puppies__ddo_cnt_df['Operation_Type'].value_counts())
# ----------------------------------------------------
# Operation_Type
# Adoption         2048
# Transfer Out      565
# Euthanasia         53
# Died               20
# Missing             3
# Clinic Out          1
# Admin Missing       1
# Name: count, dtype: int64


per_dog__no_puppies__ddo_cnt_df = per_dog__no_puppies__ddo_cnt_df.reset_index(drop=True)



# --------------------------------------------------
# One hot encoding
# --------------------------------------------------
per_dog__no_puppies__ddo_cnt_df_encoded = pd.get_dummies(per_dog__no_puppies__ddo_cnt_df, #drop_first=True,
                                                         columns=['Primary_Breed', 'Gender', 'Primary_Colour', 'Age_Group','Condition_shifted','Size_shifted'])



# -----------------------------------------------------------------------------
# X & y DATA!!!! ***
# -----------------------------------------------------------------------------
X = per_dog__no_puppies__ddo_cnt_df_encoded.drop(['Operation_Type'], axis=1)


# -----------------------
# correlation_matrix of X
# -----------------------
correlation_matrix = X.corr()
plt.figure(figsize=(18, 16))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool)) # Masking the Upper Triangle
sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap='bwr',
    annot=False,        
    linewidths=.5,
    cbar_kws={'shrink': 0.8, 'format': '%.2f'}
)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()



y = per_dog__no_puppies__ddo_cnt_df['Operation_Type']
# print(y.value_counts())


# -----------------------
# Create a binary target ***
# -----------------------
y = y.apply(lambda x: 1 if x == 'Adoption' else 0)

y.value_counts()
# Operation_Type
# 1    2048
# 0     643
# Name: count, dtype: int64


# --------------------------------------------------
# use MinMaxScaler  to scale numerical features to a specific range, typically between 0 and 1.
# --------------------------------------------------

# -------------------------------
# Scale all of X data to run PCA
# -------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)



# -------------------------------
# Scale for model creation
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=project_seed)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# -----------------------
# create X w/o DDO
# -----------------------
X_wo_DDO = X.drop(['num_of_DDO'], axis=1)


X_train_wo_DDO, X_test_wo_DDO, y_train_wo_DDO, y_test_wo_DDO = train_test_split(X_wo_DDO, y, test_size=0.20, random_state=project_seed)


scaler_wo_DDO = MinMaxScaler()
X_train_wo_DDO = scaler_wo_DDO.fit_transform(X_train_wo_DDO)
X_test_wo_DDO = scaler_wo_DDO.transform(X_test_wo_DDO)



# -----------------------------------------------------------------------------
# RESAMPLE X & y DATA!!!! ***
# -----------------------------------------------------------------------------

# print("Original class distribution:", Counter(y))

# Oversampling using RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='not majority', random_state=project_seed)
X_over, y_over = oversample.fit_resample(X, y)
# print("Oversampled class distribution:", Counter(y_over))


# Undersampling using RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='not minority', random_state=project_seed)
X_under, y_under = undersample.fit_resample(X, y)
# print("Undersampled class distribution:", Counter(y_under))


# -------------------------------

X_train_over, X_test_over, y_train_over, y_test_over = train_test_split(X, y, test_size=0.20, random_state=project_seed)

scaler_over = MinMaxScaler()
X_train_over = scaler_over.fit_transform(X_train_over)
X_test_over = scaler_over.transform(X_test_over)
# print("Original class distribution:", Counter(y_train_over))


# Oversampling using RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='not majority', random_state=project_seed)
X_train_over, y_train_over = oversample.fit_resample(X_train_over, y_train_over)
# print("Oversampled class distribution:", Counter(y_train_over))




X_train_under, X_test_under, y_train_under, y_test_under = train_test_split(X, y, test_size=0.20, random_state=project_seed)

scaler_under = MinMaxScaler()
X_train_under = scaler_under.fit_transform(X_train_under)
X_test_under = scaler_under.transform(X_test_under)


print("Original class distribution:", Counter(y_train_under))
# undersampling using RandomunderSampler
undersample = RandomUnderSampler(sampling_strategy='not minority', random_state=project_seed)
X_train_under, y_train_under = undersample.fit_resample(X_train_under, y_train_under)
print("undersampled class distribution:", Counter(y_train_under))



# -----------------------------------------------------------------------------
# Mutual information ***
# -----------------------------------------------------------------------------
mi_scores = mutual_info_classif(X, y, random_state=project_seed)
# mi_scores = mutual_info_classif(X_scale, y, random_state=project_seed) # results are the same with scaled or unscaled data

# Display scores
# print(f"Mutual Information scores: {mi_scores}")
# for col, MI in zip(X.columns, mi_scores):
#     print(f"Feature: {col}, mutual information: {MI}")
    
    
mi_scores_df = pd.DataFrame(zip(X.columns, mi_scores), columns=['Feature', 'MI_Score'])
mi_scores_df = mi_scores_df.sort_values(by='MI_Score', ascending=False) # Sort by 'MI_Score' 
mi_scores_df = mi_scores_df.reset_index(drop=True)



plt.figure(figsize=(10, 6))
bars = plt.barh(mi_scores_df['Feature'], mi_scores_df['MI_Score'], color='skyblue')

plt.xlabel('MI_Score')
plt.ylabel('Feature')
plt.title('Mutual Information Bar Plot')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()




# ------------------------------------------
# Mutual information  Oversampled data
# ------------------------------------------
mi_scores_over = mutual_info_classif(X_over, y_over, random_state=project_seed)
    
    
mi_scores_over_df = pd.DataFrame(zip(X_over.columns, mi_scores_over), columns=['Feature', 'MI_Score'])
mi_scores_over_df = mi_scores_over_df.sort_values(by='MI_Score', ascending=False) # Sort by 'MI_Score' 
mi_scores_over_df = mi_scores_over_df.reset_index(drop=True)



# ------------------------------------------
# Mutual information  Undersampled data
# ------------------------------------------
mi_scores_under = mutual_info_classif(X_under, y_under, random_state=project_seed)
    
    
mi_scores_under_df = pd.DataFrame(zip(X_under.columns, mi_scores_under), columns=['Feature', 'MI_Score'])
mi_scores_under_df = mi_scores_under_df.sort_values(by='MI_Score', ascending=False) # Sort by 'MI_Score' 
mi_scores_under_df = mi_scores_under_df.reset_index(drop=True)




# ---------------------------------------------------
# Select top k features by Mutual information
# ---------------------------------------------------
# X.shape # (3657, 97)
# OG_feature_dim = X.shape[1]

top_k = 20
selected_features = mi_scores_df.head(top_k)
feature_liz = list(selected_features['Feature'])




plt.figure(figsize=(10, 6))
bars = plt.barh(selected_features['Feature'], selected_features['MI_Score'], color='skyblue')

max_width = selected_features['MI_Score'].max()
plt.xlim(0, max_width * 1.15)  # Extend axis

for bar in bars:
    width = bar.get_width()
    plt.text(width+ 0.005,   
             bar.get_y() + bar.get_height()/2,
             f'{width:.4f}', 
             va='center', ha='center', fontsize=9, color='black')
    
plt.xlabel('MI_Score')
plt.ylabel('Feature')
plt.title('Top 20 Mutual Information Bar Plot')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()





X_selected = X.loc[:, feature_liz]

scaler = MinMaxScaler()
X_scaled_selected = scaler.fit_transform(X_selected)


X_train_selected, X_test_selected, y_train_selected, y_test_selected = train_test_split(X_selected, y, test_size=0.20, random_state=project_seed)


scaler_selected = MinMaxScaler()
X_train_selected = scaler_selected.fit_transform(X_train_selected)
X_test_selected = scaler_selected.transform(X_test_selected)


# ----------------------------------
# top k features by MI W/O DDO
# ----------------------------------
X_selected_wo_DDO = X_selected.drop(['num_of_DDO'], axis=1)


X_train_selected_wo_DDO, X_test_selected_wo_DDO, y_train_selected_wo_DDO, y_test_selected_wo_DDO = train_test_split(X_selected_wo_DDO, y, test_size=0.20, random_state=project_seed)


scaler_selected_wo_DDO = MinMaxScaler()
X_train_selected_wo_DDO = scaler_selected_wo_DDO.fit_transform(X_train_selected_wo_DDO)
X_test_selected_wo_DDO = scaler_selected_wo_DDO.transform(X_test_selected_wo_DDO)



# ----------------------------------
# correlation_matrix of X_selected
# ----------------------------------
correlation_matrix = X_selected.corr()
plt.figure(figsize=(18, 16))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool)) # Masking the Upper Triangle
sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap='bwr',
    annot=False,        
    linewidths=.5,
    cbar_kws={'shrink': 0.8, 'format': '%.2f'}
)
plt.xticks(rotation=90, fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.title('Correlation Matrix Heatmap of Top 20 Features', fontsize=24)
plt.tight_layout()
plt.show()


# With values
# ----------------------------------
plt.figure(figsize=(18, 16))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Masking the Upper Triangle
sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap='bwr',
    annot=True,         # This displays the correlation values
    fmt='.2f',          # Two decimal places
    linewidths=.5,
    cbar_kws={'shrink': 0.8, 'format': '%.2f'}
)
plt.xticks(rotation=90, fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.title('Correlation Matrix Heatmap of Top 20 Features', fontsize=24)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# PCA ***
# -----------------------------------------------------------------------------
# use PCA to reduce the dimensionality of the data
# Reference used: https://www.geeksforgeeks.org/implementing-pca-in-python-with-scikit-learn/


# ----------------------------------
# 2D plot!!!!
# ----------------------------------

random.seed(project_seed)
# X_scaled.shape # X_scaled.shape # X {array-like, sparse matrix} of shape (n_samples, n_features)
pca_2 = PCA(n_components=2)

X_pca_2 =pca_2.fit_transform(X_scaled) # Returns: ndarray of shape (n_samples, n_components)
# X_pca_2.shape # (n_samples, 2)



colors = plt.cm.tab10.colors

for i, item in enumerate(y.unique()):
    cluster_x = X_pca_2[y==item,0]
    cluster_y = X_pca_2[y==item,1]
    plt.scatter(cluster_x, cluster_y, color=colors[i], label=f'{item}', alpha=0.8)
    
    
legend_labels = {0: 'Other', 1: 'Adopted'}
handles, labels = plt.gca().get_legend_handles_labels()
custom_labels = [legend_labels[int(lbl)] for lbl in labels]
plt.legend(handles, custom_labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.title('DDO Binary Target - PCA(n_components=2)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


PC_coefficients = pca_2.components_
# print(PC_coefficients)

# Create PC_coefficients DataFrame 
component_names = [f'PC{i+1}' for i in range(PC_coefficients.shape[0])]
feature_names = X.columns 

PC_coefficients_df2 = pd.DataFrame(PC_coefficients, index=component_names, columns=feature_names)
# print(PC_coefficients_df2)


# # absolute value to see which features contribute most to PC
# PC_coefficients_df2_abs = PC_coefficients_df2.abs().transpose().sort_values(by='PC2', ascending=False)
# PC_coefficients_df2_abs = PC_coefficients_df2.transpose().sort_values(by='PC2', ascending=False)


# ----------------------------------
# lets get some eigen values!!!!! to find out how many PCs we need! 
# ----------------------------------
random.seed(project_seed)
# Initialize and fit PCA
pca = PCA()
pca.fit(X)

# eigenvalues!!!
eigenvalues = pca.explained_variance_

explained_variance_ratio = pca.explained_variance_ratio_

# print("Eigenvalues (explained variance):")
# print(eigenvalues)


# print("explained_variance_ratio:")
# print(explained_variance_ratio)

# print("\nCumulative Sum of explained_variance_ratio:")
# print(np.cumsum(explained_variance_ratio))

# 1 PC gives us 0.99993209 of explained_variance_ratio
# ----------------------------------------------------
# Almost all the variability in the dataset can be captured by a single linear combination of features.
# The first principal component provides an excellent summary of the data, and additional components add very little extra information.




plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-')
plt.title('Explained Variance Ratio per Principal Component')
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.xticks(range(1, len(explained_variance_ratio) + 1)) # Ensure integer ticks for components
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), marker='o', linestyle='-')
plt.title('Cumulative Explained Variance Ratio')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.grid(True)
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.show()



# ----------------------------------
# PC_coefficients
# ----------------------------------
PC_coefficients = pca.components_
# print(PC_coefficients)

# Create PC_coefficients DataFrame 
component_names = [f'PC{i+1}' for i in range(PC_coefficients.shape[0])]
feature_names = X.columns 

PC_coefficients_df = pd.DataFrame(PC_coefficients, index=component_names, columns=feature_names)
# print(PC_coefficients_df)


# absolute value to see which features contribute most to PC
PC_coefficients_df_abs = PC_coefficients_df.abs().transpose().sort_values(by='PC2', ascending=False)
PC_coefficients_df_abs = PC_coefficients_df.transpose().sort_values(by='PC2', ascending=False)



# ----------------------------------
# PCA in Train and test
# ----------------------------------
pca_for_modeling = PCA(n_components=1)
X_train_pca = pca_for_modeling.fit_transform(X_train)
X_test_pca = pca_for_modeling.transform(X_test)


# -----------------------------------------------------------------------------
# Random Forest ***
# -----------------------------------------------------------------------------
# -------------------------------------
# Random Forest ALL THE Features
# -------------------------------------
rf_search_df, rf_feature_scores_df  = rf_search(X, X_train, X_test, y_train, y_test, project_seed)



# -------------------------------------
# SVM W/O DDO ALL THE Features
# -------------------------------------
rf_search_wo_DDO_df, rf_feature_scores_wo_DDO_df  = rf_search(X, X_train_wo_DDO, X_test_wo_DDO, y_train_wo_DDO, y_test_wo_DDO, project_seed)



# -------------------------------------
# Random Forest  TOP MI Features
# -------------------------------------
rf_search_top_selected_df, rf_feature_scores_top_selected_df = rf_search(X_selected, X_train_selected, X_test_selected, y_train_selected, y_test_selected, project_seed)


# -------------------------------------
# Random Forest  TOP MI Features W/O DDO
# -------------------------------------
rf_search_top_selected_wo_DDO_df, rf_feature_scores_top_selected_wo_DDO_df = rf_search(X_selected_wo_DDO, X_train_selected_wo_DDO, X_test_selected_wo_DDO, y_train_selected_wo_DDO, y_test_selected_wo_DDO, project_seed)


# -------------------------------------
# Random Forest w/ PCA data >>>> lower scores for PCA data
# -------------------------------------
rf_search_PCA_df, rf_feature_scores_PCA_df  = rf_search(X, X_train_pca, X_test_pca, y_train, y_test, project_seed)



# -------------------------------------
# Random Forest w/ Oversampled data 
# -------------------------------------
rf_search_over_df, rf_feature_scores_over_df  = rf_search(X_over, X_train_over, X_test_over, y_train_over, y_test_over, project_seed)



# -------------------------------------
# Random Forest w/ Undersampled data 
# -------------------------------------
rf_search_under_df, rf_feature_scores_under_df  = rf_search(X_under, X_train_under, X_test_under, y_train_under, y_test_under, project_seed)




# -------------------------------------------------------
# print_classification_report
# -------------------------------------------------------

# print_classification_report("RFC", rf_search_df, "RF using all X features", X_train, X_test, y_train, y_test)

# print_classification_report("RFC", rf_search_wo_DDO_df, "RF: all X features w/o DDO", X_train_wo_DDO, X_test_wo_DDO, y_train_wo_DDO, y_test_wo_DDO)

# print_classification_report("RFC", rf_search_top_selected_df, "RF: TOP MI X features", X_train_selected, X_test_selected, y_train_selected, y_test_selected)

# print_classification_report("RFC", rf_search_top_selected_wo_DDO_df, "RF: TOP MI X features w/o DDO", X_train_selected_wo_DDO, X_test_selected_wo_DDO, y_train_selected_wo_DDO, y_test_selected_wo_DDO)

# print_classification_report("RFC", rf_search_PCA_df, "RF: PCA data", X_train_pca, X_test_pca, y_train, y_test)

# print_classification_report("RFC", rf_search_over_df, "RF: Oversampled data", X_train_over, X_test_over, y_train_over, y_test_over)

# print_classification_report("RFC", rf_search_under_df, "RF: Undersampled data", X_train_under, X_test_under, y_train_under, y_test_under)



# -------------------------------------------------------
# create  rf_top_models_CV_df  - based on CV results to allow me to rank models
# -------------------------------------------------------

rf_df_liz = [rf_search_df,rf_search_wo_DDO_df,rf_search_top_selected_df,rf_search_top_selected_wo_DDO_df,rf_search_PCA_df,rf_search_over_df,rf_search_under_df]
rf_df_liz_nmae = ['rf_search_df','rf_search_wo_DDO_df','rf_search_top_selected_df','rf_search_top_selected_wo_DDO_df','rf_search_PCA_df','rf_search_over_df','rf_search_under_df']

rf_top_models_CV = []
for name, df in zip(rf_df_liz_nmae, rf_df_liz):
    first_row = df.iloc[0].copy()
    first_row['run'] = name
    rf_top_models_CV.append(first_row)



scoring_metric_I_care_about = 'mean_test_f1_macro'
# https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
# https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/
# 'weighted' - favouring the majority class
# 'micro' - no favouring any class in particular.
# 'macro'  -  bigger penalisation when your model does not perform well with the minority classes.
    
# ['index', 'params', 'mean_test_accuracy', 'mean_test_f1_macro',
#        'mean_test_f1_micro', 'mean_test_f1_weighted']
rf_top_models_CV_df = pd.DataFrame(rf_top_models_CV)
rf_top_models_CV_df = rf_top_models_CV_df.sort_values(by=scoring_metric_I_care_about, ascending=False) 
rf_top_models_CV_df = rf_top_models_CV_df.reset_index(drop=True)



# -------------------------------------------------------
# create  rf_top_models_test_df - based on results from testing data to allow me to rank models
# -------------------------------------------------------


rf_top_models_test_data = []


rf_top_models_test_data.append(top_models_based_on_test_data("RFC", rf_search_df, "RF using all X features", X_train, X_test, y_train, y_test))

rf_top_models_test_data.append(top_models_based_on_test_data("RFC", rf_search_wo_DDO_df, "RF: all X features w/o DDO", X_train_wo_DDO, X_test_wo_DDO, y_train_wo_DDO, y_test_wo_DDO))

rf_top_models_test_data.append(top_models_based_on_test_data("RFC", rf_search_top_selected_df, "RF: TOP MI X features", X_train_selected, X_test_selected, y_train_selected, y_test_selected))

rf_top_models_test_data.append(top_models_based_on_test_data("RFC", rf_search_top_selected_wo_DDO_df, "RF: TOP MI X features w/o DDO", X_train_selected_wo_DDO, X_test_selected_wo_DDO, y_train_selected_wo_DDO, y_test_selected_wo_DDO))

rf_top_models_test_data.append(top_models_based_on_test_data("RFC", rf_search_PCA_df, "RF: PCA data", X_train_pca, X_test_pca, y_train, y_test))

rf_top_models_test_data.append(top_models_based_on_test_data("RFC", rf_search_over_df, "RF: Oversampled data", X_train_over, X_test_over, y_train_over, y_test_over))

rf_top_models_test_data.append(top_models_based_on_test_data("RFC", rf_search_under_df, "RF: Undersampled data", X_train_under, X_test_under, y_train_under, y_test_under))



scoring_metric_I_care_about_test = 'f1_macro'
rf_top_models_test_df = pd.concat(rf_top_models_test_data)
rf_top_models_test_df = rf_top_models_test_df.sort_values(by=scoring_metric_I_care_about_test, ascending=False) 
rf_top_models_test_df = rf_top_models_test_df.reset_index(drop=True)




# -------------------------------------------------------
#  use top model from test data w/ DDO to get RF feature_importances
# -------------------------------------------------------

best_model_prams = rf_search_df['params'].iloc[0]

rf_best_model = RFC(**best_model_prams)
    
rf_best_model.fit(X_train, y_train)
# y_pred = rf_best_model.predict(X_test)
rf_feature_scores_best_model = rf_best_model.feature_importances_

rf_feature_scores_best_model_test_df = pd.DataFrame(zip(X.columns, rf_feature_scores_best_model), columns=['Feature', 'feature_importances'])
rf_feature_scores_best_model_test_df = rf_feature_scores_best_model_test_df.sort_values(by='feature_importances', ascending=False) 
rf_feature_scores_best_model_test_df = rf_feature_scores_best_model_test_df.reset_index(drop=True)


plt.figure(figsize=(10, 6))
bars = plt.barh(rf_feature_scores_best_model_test_df['Feature'], rf_feature_scores_best_model_test_df['feature_importances'], color='skyblue')

plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importances Bar Plot')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()


# Plot top 10
#----------------------

top_x = rf_feature_scores_best_model_test_df.sort_values(by='feature_importances', ascending=False).head(15)

plt.figure(figsize=(10, 6))
bars = plt.barh(top_x['Feature'], top_x['feature_importances'], color='skyblue')


max_width = top_x['feature_importances'].max()
plt.xlim(0, max_width * 1.1)  # Extend axis


for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.02, 
             bar.get_y() + bar.get_height() / 2,
             f'{width:.3f}',
             va='center', ha='center',
             fontsize=10, color='black')

plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('RF All Features w/ DDO Model Feature Importance')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()



# -----------------------------------------------------------------------------
# SVM *** 
# -----------------------------------------------------------------------------
# -------------------------------------
# SVM  ALL THE Features
# -------------------------------------
SVM_search_df = svm_search(X_train, X_test, y_train, y_test)


# -------------------------------------
# SVM W/O DDO ALL THE Features
# -------------------------------------
SVM_search_wo_DDO_df = svm_search(X_train_wo_DDO, X_test_wo_DDO, y_train_wo_DDO, y_test_wo_DDO)


# -------------------------------------
# SVM  TOP MI Features
# -------------------------------------
SVM_search_top_selected_df = svm_search(X_train_selected, X_test_selected, y_train_selected, y_test_selected)


# -------------------------------------
# Random Forest  TOP MI Features W/O DDO
# -------------------------------------
SVM_search_top_selected_wo_DDO_df  = svm_search(X_train_selected_wo_DDO, X_test_selected_wo_DDO, y_train_selected_wo_DDO, y_test_selected_wo_DDO)



# -------------------------------------
# SVM  w/ PCA data >>>> lower scores for PCA data
# -------------------------------------
SVM_search_PCA_df = svm_search(X_train_pca, X_test_pca, y_train, y_test)



# -------------------------------------
# SVM w/ Oversampled data 
# -------------------------------------
SVM_search_over_df  = svm_search(X_train_over, X_test_over, y_train_over, y_test_over)



# -------------------------------------
# SVM w/ Undersampled data 
# -------------------------------------
SVM_search_under_df  = svm_search(X_train_under, X_test_under, y_train_under, y_test_under)




# -------------------------------------------------------
# print_classification_report
# -------------------------------------------------------

# print_classification_report("SVM", SVM_search_df, "SVM using all X features", X_train, X_test, y_train, y_test)

# print_classification_report("SVM", SVM_search_wo_DDO_df, "SVM: all X features w/o DDO", X_train_wo_DDO, X_test_wo_DDO, y_train_wo_DDO, y_test_wo_DDO)

# print_classification_report("SVM", SVM_search_top_selected_df, "SVM: TOP MI X features", X_train_selected, X_test_selected, y_train_selected, y_test_selected)

# print_classification_report("SVM", SVM_search_top_selected_wo_DDO_df, "SVM: TOP MI X features w/o DDO", X_train_selected_wo_DDO, X_test_selected_wo_DDO, y_train_selected_wo_DDO, y_test_selected_wo_DDO)

# print_classification_report("SVM", SVM_search_PCA_df, "SVM: PCA data", X_train_pca, X_test_pca, y_train, y_test)

# print_classification_report("SVM", SVM_search_over_df, "SVM: Oversampled data", X_train_over, X_test_over, y_train_over, y_test_over)

# print_classification_report("SVM", SVM_search_under_df, "SVM: Undersampled data", X_train_under, X_test_under, y_train_under, y_test_under)



# -------------------------------------------------------
# create  SVM_top_models_CV_df - based on CV results to allow me to rank models
# -------------------------------------------------------

SVM_df_liz = [SVM_search_df,SVM_search_wo_DDO_df,SVM_search_top_selected_df,SVM_search_top_selected_wo_DDO_df,SVM_search_PCA_df,SVM_search_over_df,SVM_search_under_df]
SVM_df_liz_nmae = ['SVM_search_df','SVM_search_wo_DDO_df','SVM_search_top_selected_df','SVM_search_top_selected_wo_DDO_df','SVM_search_PCA_df','SVM_search_over_df','SVM_search_under_df']

SVM_top_models_CV = []
for name, df in zip(SVM_df_liz_nmae, SVM_df_liz):
    first_row = df.iloc[0].copy()
    first_row['run'] = name
    SVM_top_models_CV.append(first_row)


# scoring_metric_I_care_about = 'mean_test_f1_macro'
# https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult
# https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f/
# 'weighted' - favouring the majority class
# 'micro' - no favouring any class in particular.
# 'macro'  -  bigger penalisation when your model does not perform well with the minority classes.
    
# ['index', 'params', 'mean_test_accuracy', 'mean_test_f1_macro',
#        'mean_test_f1_micro', 'mean_test_f1_weighted']

SVM_top_models_CV_df = pd.DataFrame(SVM_top_models_CV)
SVM_top_models_CV_df = SVM_top_models_CV_df.sort_values(by=scoring_metric_I_care_about, ascending=False) 
SVM_top_models_CV_df = SVM_top_models_CV_df.reset_index(drop=True)



# -------------------------------------------------------
# create  SVM_top_models_test_df - based on results from testing data to allow me to rank models
# -------------------------------------------------------

SVM_top_models_test_data = []


SVM_top_models_test_data.append(top_models_based_on_test_data("SVM", SVM_search_df, "SVM using all X features", X_train, X_test, y_train, y_test))

SVM_top_models_test_data.append(top_models_based_on_test_data("SVM", SVM_search_wo_DDO_df, "SVM: all X features w/o DDO", X_train_wo_DDO, X_test_wo_DDO, y_train_wo_DDO, y_test_wo_DDO))

SVM_top_models_test_data.append(top_models_based_on_test_data("SVM", SVM_search_top_selected_df, "SVM: TOP MI X features", X_train_selected, X_test_selected, y_train_selected, y_test_selected))

SVM_top_models_test_data.append(top_models_based_on_test_data("SVM", SVM_search_top_selected_wo_DDO_df, "SVM: TOP MI X features w/o DDO", X_train_selected_wo_DDO, X_test_selected_wo_DDO, y_train_selected_wo_DDO, y_test_selected_wo_DDO))

SVM_top_models_test_data.append(top_models_based_on_test_data("SVM", SVM_search_PCA_df, "SVM: PCA data", X_train_pca, X_test_pca, y_train, y_test))

SVM_top_models_test_data.append(top_models_based_on_test_data("SVM", SVM_search_over_df, "SVM: Oversampled data", X_train_over, X_test_over, y_train_over, y_test_over))

SVM_top_models_test_data.append(top_models_based_on_test_data("SVM", SVM_search_under_df, "SVM: Undersampled data", X_train_under, X_test_under, y_train_under, y_test_under))



scoring_metric_I_care_about_test = 'f1_macro'
SVM_top_models_test_df = pd.concat(SVM_top_models_test_data)
SVM_top_models_test_df = SVM_top_models_test_df.sort_values(by=scoring_metric_I_care_about_test, ascending=False) 
SVM_top_models_test_df = SVM_top_models_test_df.reset_index(drop=True)



# -------------------------------------------------------
# use top model from test data w/ DDO to get coefficents from linear SVM for feature_importances
# -------------------------------------------------------

best_model_prams = SVM_search_top_selected_df['params'].iloc[0]

SVM_best_model = SVC(**best_model_prams)
    
SVM_best_model.fit(X_train_selected, y_train_selected)
# y_pred = SVM_best_model.predict(X_test_selected)
SVM_feature_scores_best_model = SVM_best_model.coef_[0]

SVM_feature_scores_best_model_test_df = pd.DataFrame(zip(X_selected.columns, SVM_feature_scores_best_model), columns=['Feature', 'feature_importances'])
SVM_feature_scores_best_model_test_df['abs_feature_importances'] = SVM_feature_scores_best_model_test_df['feature_importances'].abs()
SVM_feature_scores_best_model_test_df = SVM_feature_scores_best_model_test_df.sort_values(by='abs_feature_importances', ascending=False)
SVM_feature_scores_best_model_test_df = SVM_feature_scores_best_model_test_df.reset_index(drop=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(SVM_feature_scores_best_model_test_df['Feature'], SVM_feature_scores_best_model_test_df['abs_feature_importances'], color='skyblue')

plt.xlabel('Absolute Value of SVM Coefficient')
plt.ylabel('Feature')
plt.title('SVM Absolute Value Coefficient Bar Plot')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.tight_layout()
plt.show()

# --------------------

SVM_feature_scores_best_model_test_df = SVM_feature_scores_best_model_test_df.sort_values(by='feature_importances', ascending=False)
plt.figure(figsize=(10, 6))
bars = plt.barh(SVM_feature_scores_best_model_test_df['Feature'], SVM_feature_scores_best_model_test_df['feature_importances'], color='skyblue')

# Add values inside the bars
max_width = SVM_feature_scores_best_model_test_df['feature_importances'].max()
for bar in bars:
    width = bar.get_width()
    plt.text(width / 2,
             bar.get_y() + bar.get_height() / 2,
             f'{width:.3f}',
             va='center', ha='center', fontsize=9, color='black')

plt.xlabel('SVM Coefficients')
plt.ylabel('Feature')
plt.title('SVM Top 20 Mutual Information Feature Model Coefficients')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



