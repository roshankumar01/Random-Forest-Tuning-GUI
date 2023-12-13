# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.datasets import load_diabetes , load_boston


st.set_page_config(page_title="ML App" , layout="wide")


def build_model(df):
    x = df.iloc[:,:-1]
    y =df.iloc[:,-1]
    
    st.markdown('Data splits')
    st.write('Training set')
    st.info(x.shape)
    st.write('test set')
    st.info(y.shape)
    
    st.markdown('variable details')
    st.write('x variable')
    st.info(list(x.columns))
    st.write('y variable')
    st.info(y.name)
    
  
    x_train , x_test, y_train, y_test =  train_test_split(x,y,test_size=split_size)
    
    rf = RandomForestRegressor(n_estimators=parameter_n_estimator , 
         random_state = paramter_random_state ,
         max_features = paramter_max_features,
         criterion= paramter_criterion ,
         min_samples_split = paramter_min_samples_split ,
         min_samples_leaf = paramter_min_samples_leaf ,
         bootstrap = paramter_bootstrap,
         oob_score = paramter_oob_score ,
         n_jobs = paramter_n_jobs)
    
    rf.fit(x_train,y_train)
    
    st.subheader( ' 2 - model performance')
    
    st.markdown('training set')
    y_pred_train = rf.predict(x_train)
    
    st.write(" R2 score")
    st.info(r2_score(y_train,y_pred_train))
    
    st.write(" MSE ")
    st.info(mean_squared_error(y_train,y_pred_train))
    
    
    
    st.markdown('test set')
    y_pred_test = rf.predict(x_test)
    
    st.write(" R2 score")
    st.info(r2_score(y_test,y_pred_test))
    
    st.write(" MSE ")
    st.info(mean_squared_error(y_test,y_pred_test))
    
    st.subheader("3 - model parameters")
    st.write(rf.get_params())
    
st.write("ML Apps")

with st.sidebar.header('1 - upload your CSV file'):
    uploaded_file = st.sidebar.file_uploader('upload your data' )

with st.sidebar.header('2 - Set Parameters'):
    split_size = st.sidebar.slider('Data split ratio (% for training dataset',10,90,80,5)
    
with st.sidebar.subheader('2.1 - Learning parameters'):
    parameter_n_estimator = st.sidebar.slider('no of estimators',0,1000,100,50)
    paramter_max_features = st.sidebar.select_slider('max features',options = ['auto','sqrt','log2'])
    paramter_min_samples_split = st.sidebar.slider('min no of split',1,10,2,1)
    paramter_min_samples_leaf = st.sidebar.slider('min no of sample for leaf node',1,10,2,1)

with st.sidebar.subheader('2.2 - General Parameter'):
    paramter_random_state = st.sidebar.slider('Random state',0,100,42,1)
    paramter_criterion = st.sidebar.select_slider('Error measure',options=['mse','mae'])
    paramter_bootstrap = st.sidebar.select_slider('bootstrap',options=[True,False])
    paramter_oob_score = st.sidebar.select_slider('out of box score',options=[True,False])
    paramter_n_jobs = st.sidebar.select_slider('Run in parallel',options=[1,-1])
    
    
st.subheader('1 - Dataset')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('1.1 - Glimpse of dataset')
    st.write(df)
    build_model(df)
    
else:
    st.info('Awaiting for CSV file to be uploaded')
    if st.button('Press to use Example Dataset'):
        
        
        boston = load_boston()
        x=pd.DataFrame(boston.data,columns=boston.feature_names)
        y=pd.Series(boston.target, name='response')
        df=pd.concat([x,y],axis=1)
        
        st.markdown('The Boston housing dataset')
        st.write(df.head(5))
        
        build_model(df)
        
