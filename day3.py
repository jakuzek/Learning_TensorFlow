from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np                                  # A very optimized version of arrays in Python (allows us to do multi dimensional calculations)
import pandas as pd                                 # It's kind of a data analystics tool, allows us to easily manipulate data like loadking data sets, view data sets etc
import matplotlib.pyplot as plt                     # It's a visualization, graph and charts

from tensorflow import feature_column as fc

import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')   # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')     # testing data
dftrain.head()                              # Shows us the first five entries in our data set
y_train = dftrain.pop('survived')           # Takes entire 'survived' column and removes it from 'dftrain' data frame and stores it in the variable y_train
y_eval = dfeval.pop('survived')             # Same as above
dftrain.loc[0], y_train.loc[0]              # If we want to find one specific row in our data frame we use .loc[row-1] (y_train.loc[0] return 0/1 in 'object' which stands for surviving)
dftrain['age']                              # If we want to find one specific column we just use ['column_name']

dftrain.describe()                          # If we want some infomation about our data set we use .describe()
dftrain.shape                               # Tells us how many rows/entries and column/features we have


# Histogram of Age
def Histogram_of_Age():
    # dftrain.age.hist(bins=20)                             # This gives us a histogram of the age
    plt.hist(dftrain['age'], bins=20, edgecolor='black')
    plt.title('Histogram of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# Distribution of Gender
def Distribution_of_Gender():
    #dftrain.sex.value_counts().plot(kind='barh')
    dftrain['sex'].value_counts().plot(kind='barh', color='skyblue')
    plt.title('Distribution of Gender')
    plt.xlabel('Count')
    plt.ylabel('Gender')
    plt.grid(axis='x')
    plt.show()


# Distribution of Classes
def Distribution_of_Classes():
    #dftrain['class'].value_counts().plot(kind='barh')
    dftrain['class'].value_counts().plot(kind='barh', color='lightcoral')
    plt.title('Distribution of Classes')
    plt.xlabel('Count')
    plt.ylabel('Class')
    plt.grid(axis='x')
    plt.show()


# Survival Rate by Gender
def Survival_Rate_by_Gender():
    #pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
    df_concat = pd.concat([dftrain, y_train], axis=1)
    df_concat.groupby('sex')['survived'].mean().plot(kind='barh', color='lightgreen')
    plt.title('Survival Rate by Gender')
    plt.xlabel('% Survive')
    plt.ylabel('Gender')
    plt.grid(axis='x')
    plt.show()

#Histogram_of_Age()
#Distribution_of_Gender()
#Distribution_of_Classes()
#Survival_Rate_by_Gender()

# After analyzing this information we should notice the following:
# - The majority of passengers are in their 20's or 30's
# - The majority of passengers are male
# - The majority of passengers are in 'Third' class
# - Females have a much higher chance of survival


# Training vs Testing Data
print(dfeval.shape)