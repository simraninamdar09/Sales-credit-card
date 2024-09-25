#!/usr/bin/env python
# coding: utf-8
#import Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import streamlit as st
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import pickle
warnings.filterwarnings("ignore")
st.title("Sales Credit Card")
# Load the dataset
data=pd.read_csv("credit_card.csv",low_memory=False)

# Encoding data
label_encoder = preprocessing.LabelEncoder()
data['Card_Category']= label_encoder.fit_transform(data['Card_Category']) 
data['Qtr']= label_encoder.fit_transform(data['Qtr'])
data['Use Chip']= label_encoder.fit_transform(data['Use Chip']) 
data['Exp Type']= label_encoder.fit_transform(data['Exp Type']) 


data['revenue'] = data['Annual_Fees']+data['Total_Trans_Amt'] + data['Interest_Earned']


#Drop col
col_to_drop = ['Client_Num','Week_Num','Week_Start_Date','Interest_Earned','Total_Trans_Vol','Annual_Fees','current_year',
               'Total_Trans_Amt','Qtr','Use Chip']
data.drop(columns=col_to_drop,inplace=True)

# Outlier  Detection and treatment onRevenue and Credit Limit column
# Treat outliers using IQR method-credit limit
plt.figure(figsize=(10,6))
q1 = data['Credit_Limit'].quantile(0.25)
q3 = data['Credit_Limit'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data['credit_limit'] = data['Credit_Limit'].clip(lower=lower_bound, upper=upper_bound)

# Treat outliers using IQR method-Revenue
plt.figure(figsize=(10,6))
q1 = data['revenue'].quantile(0.25)
q3 = data['revenue'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data['Revenue'] = data['revenue'].clip(lower=lower_bound, upper=upper_bound)

# Delect column -Revenue, Credit limit
col_to_drop = ['Credit_Limit','revenue']
data.drop(columns=col_to_drop)

#Index fixing column Revenue
revenue_col = data.pop('Revenue')
credit_limit_col = data.pop('credit_limit')
data.insert(data.columns.get_loc('Delinquent_Acc'), 'Revenue', revenue_col)
data.insert(data.columns.get_loc('Total_Revolving_Bal'), 'credit_limit', credit_limit_col)

# feature Scaling of 'credit_limit','Revenue','Avg_Utilization_Ratio'
col_tarnsform = ['credit_limit', 'Revenue','Avg_Utilization_Ratio']
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
data[col_tarnsform] = scaler.fit_transform(data[col_tarnsform])

# Split the data into features (X) and target (y)
x = data.iloc[:,0:8]
y = data['Delinquent_Acc']
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create the Bagging classifier
bag_c=BaggingClassifier()
bag1 =bag_c.fit(x_train,y_train)
bag1.score(x_train,y_train)

bag1.score(x_test,y_test)


#Pickel File
filename = 'final_Bagging_model.pkl'
pickle.dump(bag_c, open(filename,'wb'))
bag_c.fit(x,y)
pk=bag_c.predict(x_test)



Card_category= st.selectbox('Card_category', data['Card_Category'].unique())
Activation_in_30_Days = st.selectbox('Activation_in_30_Days', data['Activation_30_Days'].unique())
Customer_Acquisition_Cost= st.selectbox('Customer_Acquisition_Cost', data['Customer_Acq_Cost'].unique())
Credit_Limit = st.selectbox('Credit_Limit', data['credit_limit'].unique())
Total_Revolving_Balance= st.selectbox('Total_Revolving_Balance', data['Total_Revolving_Bal'].unique())
Average_Utilization_Ratio = st.selectbox('Average_Utilization_Ratio', data['Avg_Utilization_Ratio'].unique())
Expense_Type = st.selectbox('Expense_Type', data['Exp Type'].unique())
Revenue = st.selectbox('Revenue', data['Revenue'].unique())



if st.button('Prevention Type'):
    df = {
        'Card_Category':Card_category,
        'Activation_30_Days':Activation_in_30_Days,
        'Customer_Acq_Cost': Customer_Acquisition_Cost,
        'credit_limit': Credit_Limit,
        'Total_Revolving_Bal': Total_Revolving_Balance,
        'Avg_Utilization_Ratio': Average_Utilization_Ratio,
        'Exp Type':Expense_Type,
        'Revenue':Revenue
    }

    df1 = pd.DataFrame(df, index=[1])
    predictions = bag_c.predict(df1)

    if predictions.any() == 1:
        prediction_value = 'Delinquent'
    else:
        prediction_value = 'Not Delinquent'

    st.title("Account is " + str(prediction_value))


    





