#!/usr/bin/env python
# coding: utf-8

import pickle
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load Data
filepath = '/Users/tim/src/Metis/Project_3/data/interim/df_merge3.pkl'
with open(filepath, 'rb') as pkl:
    df = pickle.load(pkl)

filepath = '/Users/tim/src/Metis/Project_3/data/interim/train_mean.pkl'
with open(filepath, 'rb') as pkl:
    train_mean = pickle.load(pkl)

filepath = '/Users/tim/src/Metis/Project_3/data/interim/train_median.pkl'
with open(filepath, 'rb') as pkl:
    train_median = pickle.load(pkl)
    
filepath = '/Users/tim/src/Metis/Project_3/data/interim/test_mean.pkl'
with open(filepath, 'rb') as pkl:
    test_mean = pickle.load(pkl)

# Create feature matrix (X) and target vector (y)
y = df['coup']
X = df.drop(['year', 'coup', 'country'], axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=13, stratify=y)

#Scale Data
scaler = StandardScaler()
mean_train_scaled = scaler.fit_transform(train_mean)
median_train_scaled = scaler.fit_transform(train_median)

mean_scale = scaler.fit(train_mean)
X_test_scaled = mean_scale.transform(X_test)

#Oversampling
ros = RandomOverSampler(random_state=0)
X_resampled_mean, y_resampled = ros.fit_sample(mean_train_scaled,y_train)
X_smoted_mean, y_smoted = SMOTE(random_state=42).fit_sample(mean_train_scaled,y_train)
X_adasyn_mean, y_adasyn = ADASYN(random_state=42).fit_sample(mean_train_scaled,y_train)

# X_resampled_median, y_resampled = ros.fit_sample(median_train_scaled,y_train)
# X_smoted_median, y_smoted = SMOTE(random_state=42).fit_sample(median_train_scaled,y_train)
# X_adasyn_median, y_adasyn = ADASYN(random_state=42).fit_sample(median_train_scaled,y_train)


# Record Results
def record_results(results):
    """
    Keeps track of model results.
    Input: results (dict), a dictionary of key model attributes
        Results look like:
            results = {
                'Model': 'Logistic Regression', 
                'Hyperparameters': 'solver='lbfgs', 
                'Target': 'Cancelled',
                'Features': 31,
                'Train/Test Observations': (320132, 82433),
                'Train Balance': {0: 316931, 1: 3201},
                'Train_AUC': 0.3, 
                'Test_AUC': 0.21,
                'Notes': 'MVP! Classes highly unbalanced; added weather data'
            }
    Output: record returned as a dataframe and stored as a pickle to persist
    """
    path = '~/src/Metis/Project_3/Submit/data'
    filename = 'model_history2.pkl'
    filepath = path + filename
    if not file_exists(path, filename):
        columns = ['Model', 'Hyperparameters', 'Target', 'Features', 
                   'Observations', 'Train Balance', 
                   'Train_AUC', 'CV_AUC', 'Notes']
        record = pd.DataFrame(columns=columns)
    else:
        record = pd.read_pickle(filepath)
        
    record.loc[len(record)] = results
    display(record.tail(10))
    pd.to_pickle(record, filepath)
    return record

def file_exists(path, filename):
    filepath = path + filename
    file_exists = os.path.isfile(filepath)
    return file_exists
