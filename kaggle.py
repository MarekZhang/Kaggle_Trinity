#%% md

## Import all necessary libraries

#%%

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
%matplotlib inline

#%% md

#### Handle with nan value in training set and drop features
* Drop "Hair Color" column
* Remove all rows containing nan in training set
* Remove "Wears Glasses" " Hair Color" features

#%%

Raw_Data = pd.read_csv('WithLabels.csv', index_col= 0 ) # training data  index_col = 0 
Raw_Data = Raw_Data.drop(labels = "Hair Color", axis = 1)# Drop "Hair Color" column
Raw_Data = Raw_Data.dropna(axis = 0, how = 'any')        #Revome all rows containing nan in training set
Raw_Data = Raw_Data.drop(labels = "Wears Glasses", axis = 1) #
Input = pd.read_csv('WithoutLabels.csv', index_col= 0 )
Input = Input.drop(labels = "Hair Color", axis = 1)
Input = Input.drop(labels = "Wears Glasses", axis = 1)
Input = Input.drop(labels = "Income", axis = 1)

#%% md

#### Hanle nan value in predicting set
* fill with backward and forward value for numberic features
* fill with unknown for categorical features

#%%

Input["Year of Record"].fillna(method = 'ffill', inplace = True) #将年份为nan的填为2000
Input["Age"].fillna(method = 'ffill', inplace = True) #将年龄为nan的填为55
Input["Gender"].fillna('unknown', inplace = True)
Input["Profession"].fillna('unknown', inplace = True)
Input["University Degree"].fillna('unknown', inplace = True)
#Raw_Data.isnull().sum()

#%%

#sns.scatterplot("Size of City", "Income in EUR", data = Raw_Data)

#%% md

#### Drop rows with "Size of City" value greater than 9e6

#%%

def drop_big_city(df, column, value):
    df = df.drop(df[df[column] > value].index, inplace = True)

drop_big_city(Raw_Data, "Size of City", 9000000)
    

#%%

a = Raw_Data["Size of City"]
count = 0
for i in a:
    if ( i > 9000000):
        count +=1

#sns.scatterplot("Size of City", "Income in EUR", data = Raw_Data)

#%%

#function drop rows containing specified value
def drop_unknown(df, column, value):
    df = df.drop(df[df[column] == value].index, inplace = True)
    return df

#%%

#sns_plot = sns.pairplot(Raw_Data)
#sns_plot.savefig("output.png")

#%%

#sns.catplot("University Degree", "Body Height [cm]", data = Input)

#%%

drop_unknown(Raw_Data, "Gender", '0')
drop_unknown(Raw_Data, "Gender", 'unknown')
drop_unknown(Raw_Data, "University Degree", "0")

#%% md

#### replace 0 value with unknown

#%%

def In_replace_zero(a):
    return a.replace('0', 'unknown')

Input.Gender = Input.Gender.apply(In_replace_zero)   #把input中0全部替换成unkown
Input["University Degree"] = Input["University Degree"].apply(In_replace_zero)

#%%

#sns.scatterplot(x="Year of Record", y="Body Height [cm]", data=Input);

#%%

#sns.scatterplot(x="Body Height [cm]", y="Income in EUR", data=Raw_Data);

#%% md

#### apply sqrt to the Income value

#%%

import math
def sqrt_value(df, column):
    df[column] = df[column].apply(lambda x: math.sqrt(x))
#sqrt_value(Raw_Data, "Age")

#%%

#count the Income value lower than 0
count = 0
a = Raw_Data["Income in EUR"]
for i in a:
    if(i <= 0):
        count = count + 1
        
#print(count)

#%% md

#### Revome rows with Income value lower than 0

#%%

def drop_zero_Inco(df, column):
    df = df.drop(df[df[column] < 0].index, inplace = True)
    return df
drop_zero_Inco(Raw_Data, "Income in EUR")

#%%

#sqrt the Income value
sqrt_value(Raw_Data, "Income in EUR")

#%% md

#### min-max normalization for Age feature

#%%

min_age = Raw_Data["Age"].min()
max_age = Raw_Data["Age"].max()

#%%

def nor_age_value(df, column):
    df[column] = df[column].apply(lambda x: (x-min_age)/(max_age - min_age))

nor_age_value(Raw_Data, "Age")
nor_age_value(Input, "Age")

#%%

#sns.scatterplot(x="Age", y="Income in EUR", data = Raw_Data)

#%%

#sns.scatterplot(x="Body Height [cm]", y="Income in EUR", data=Raw_Data)

#%% md

## IOR 处理前
sns.pairplot(Raw_Data)

#%% md

## Tukey IQR

#%%

def subset_by_iqr(df, column, whisker_width=1.1):
    # Calculate Q1, Q2 and IQR
    q1 = df[column].quantile(0.25)                 
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    # Apply filter with respect to IQR, including optional whiskers
    filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
    return df.loc[filter]                                                     

#%%

# Example for whiskers = 1.5, as requested by the OP
#Raw_Data = subset_by_iqr(Raw_Data, 'Income in EUR', whisker_width=1.5)
#Raw_Data = subset_by_iqr(Raw_Data, 'Year of Record', whisker_width=1.5)
#Raw_Data = subset_by_iqr(Raw_Data, 'Size of City', whisker_width=1.5)
Raw_Data = subset_by_iqr(Raw_Data, 'Body Height [cm]', whisker_width=1.5)

#sns.pairplot(Raw_Data)

#%% md

#### get the list of all unique variables in "Profession" and "Country" features both in training and predicting sets

#%%

Raw_Pro_Categ = [x for x in Raw_Data["Profession"].value_counts().sort_values(ascending = False).index]
Inp_Pro_Categ = [x for x in Input["Profession"].value_counts().sort_values(ascending = False).index]
Pro_Categ_tem = Raw_Pro_Categ + Inp_Pro_Categ #remove duplicated value
pro_Categ = list(dict.fromkeys(Pro_Categ_tem))
#len(pro_Categ)

#%%

Raw_Country_Categ = [x for x in Raw_Data["Country"].value_counts().sort_values(ascending = False).index]
Inp_Country_Categ = [x for x in Input["Country"].value_counts().sort_values(ascending = False).index]
Coun_Categ_tem = Raw_Country_Categ + Inp_Country_Categ
Country_Categ = list(dict.fromkeys(Coun_Categ_tem))

#%% md

#### function of one-hot

#%%

def one_hot_top_x(df, variable, top_x_labels):
    for label in top_x_labels:
        df[variable + '_' +label] = np.where(df[variable] == label, 1, 0)


#%%

one_hot_top_x(Raw_Data, "Profession", pro_Categ)
one_hot_top_x(Raw_Data, "Country", Country_Categ)

#%%

one_hot_top_x(Input, "Profession", pro_Categ)
one_hot_top_x(Input, "Country", Country_Categ)

#%%

Raw_Data = Raw_Data.drop(labels = 'Profession', axis = 1)
Raw_Data = Raw_Data.drop(labels = 'Country', axis = 1)

#%%

Input = Input.drop(labels = 'Profession', axis = 1)
Input = Input.drop(labels = 'Country', axis = 1)

#%% md

#### get_dummies for other categorical features ##

#%%

Data_after_dummy = pd.get_dummies(Raw_Data, columns=['Gender', 'University Degree'], drop_first=True)
Input_after_dummy = pd.get_dummies(Input, columns=['Gender', 'University Degree'], drop_first=True)

#%%

Input_after_dummy = Input_after_dummy.drop(labels = "Gender_unknown", axis = 1)
Input_after_dummy = Input_after_dummy.drop(labels = "University Degree_unknown", axis = 1)

#%% md

#### split Income column and features

#%%

Income_after_dummy = Data_after_dummy["Income in EUR"]
Data_after_dummy = Data_after_dummy.drop(labels = "Income in EUR", axis = 1)

#%% md

#### Data split and prepare for training

#%%

train_inputs, rest_inputs, train_labels, rest_labels = train_test_split(Data_after_dummy, Income_after_dummy, test_size = 0.20) #cross_validation CHANGED TO model_selection

#%% md



#%% md

## <mark>model training</mark> ##

#%%

from sklearn import linear_model
clf = linear_model.reg = linear_model.LinearRegression()
clf.fit(X=train_inputs, y=train_labels)
valid_pred = clf.predict(rest_inputs)
print(clf.coef_.shape)
mean_squared_error(rest_labels, valid_pred) # how good the model is

#%% md

#### Result Output

#%%

OutcomeFinal = clf.predict(Input_after_dummy)

#%%

pd.DataFrame(OutcomeFinal).to_csv('Predicted_Income.csv')
