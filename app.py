import streamlit as st
import pandas as pd
import shap
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt

from eda import get_rare_pie, load_data
from model import load_model, get_classification_report_to_custom_threshold, get_prediction_by_index, get_shap_values_by_index


GENERAL_DF = 'datasets/general_df.csv'
GENERAL_DF_CLEANED = 'datasets/general_df_cleaned.csv'
LOANS_DF = 'datasets/loans_df.csv'


continuous_features = ['credit', 'fst_payment', 'age', 'child_total', 'dependants',
                       'own_auto', 'personal_income', 'work_time', 'closed_loans_count',
                       'total_loans_count']

#SECTION 1
# DATA LOADING

general_df = load_data(GENERAL_DF)
general_df_cleaned = load_data(GENERAL_DF_CLEANED)
loans_df = load_data(LOANS_DF)

clients_without_loans_now = 9477
clients_with_loans_now = len(general_df) - clients_without_loans_now

#SECTION 2 PAGE: TITLE AND DESCRIPTION
st.title('Exploratory analysis of bank loan data and linear decision model training.')
st.divider()
st.markdown('Report on the preliminary analysis of data on customer loans. \n'
             'Data has been merged from multiple tables in a database. \n'
        'Here are the results of the analysis of the quality, \n'
        'completeness of the data and further opportunities to work on \n'
        'the basis of these data. \n')

st.markdown('In addition, here you can test a logistic regression model \
        trained to predict the likelihood of a customer responding to an advertising offer.')

#SECTION 2 PAGE: EDA BASIC CONCLUSIONS
st.header('EDA conclusions')
st.subheader('Basic conclusions about data and loans')
st.markdown('1. The raw combined data contained `300` duplicates, which was `2%` of all the original data.')
st.markdown('2. Count of clients without loans now is: `9477`')

loans_and_clients_fig = plt.figure(figsize=(10, 6))

plt.pie([clients_without_loans_now, clients_with_loans_now], autopct='%1.0f%%',
            colors=sns.color_palette('Set2'), shadow=True,
             wedgeprops = {"edgecolor" : "white",
                           'linewidth': 2,
                           'antialiased': True})

plt.legend([f'Clients without loans ({clients_without_loans_now})', f'Clients with loans ({clients_with_loans_now})'], bbox_to_anchor=(1, 0.6))
plt.title('Clients and loans status', weight='bold');

st.write(loans_and_clients_fig)

st.markdown('3. Repeat customers from the total number: `25.85%`')

totals_count_index = general_df['total_loans_count'].value_counts().index
totals_count_values = general_df['total_loans_count'].value_counts().values

loans_per_client_fig = plt.figure(figsize=(10, 6))
bar = plt.bar(totals_count_index, totals_count_values, alpha=0.7, color='green')

plt.bar_label(bar)
plt.xticks(range(0, 12))
plt.title('Number of loans per client', weight='bold');

st.write(loans_per_client_fig)

#SECTION 2 PAGE: EDA CONCLUSIONS ABOUT TARGET AND FEATURES QUALITY

st.subheader('Conclusions about the quality of features and target')

st.markdown('1. Target value is imbalanced.')
st.text('% of positive samples: 11.9 (1812)')
st.text('% of negative samples: 88.1 (13411)')

st.markdown('2. There may be incorrect data about the work.')
st.text('The maximum value of the years during which a person has worked in the company: 238996 (years)')
st.text('The number of people working at the current place earlier than from the age of 16: 63')

st.markdown('3. Deleting incorrect values changes the correlation between the features `age` and `work_time`')

correlation_fig, axes = plt.subplots(2, 1, figsize=(20, 15))

sns.heatmap(general_df_cleaned[continuous_features].corr(), annot=True, ax=axes[0])
axes[0].set_xticks([])
axes[0].set_title('Correlation matrix for continuous features on cleaned data.', weight='bold')

sns.heatmap(general_df[continuous_features].corr(), annot=True, ax=axes[1])
axes[1].set_title('Correlation matrix for continuous features on not cleaned data.', weight='bold')

plt.xticks(rotation=10);

st.write(correlation_fig)

st.markdown('4. In categorical data, there are rare values by category in fetures: \
            `education`, `reg, fact, postal` + `_address_province` . \n '
            'This can lead to bad generalizing ability of the model.\n'
            
        'Also  features `gen_industry`, `gen_title`, `job_dir` contains samples with empty fields.')

pie_fig_reg = get_rare_pie(general_df['postal_address_province'], 5)
st.write(pie_fig_reg)
st.text('Postal adress feature')

pie_fig_reg = get_rare_pie(general_df['reg_address_province'], 5)
st.write(pie_fig_reg)
st.text('Reg adress feature')

pie_fig_reg = get_rare_pie(general_df['fact_address_province'], 5)
st.write(pie_fig_reg)
st.text('Fact adress feature')

st.markdown('5. Features `gen_industry`, `gen_title`, `job_dir` contains samples with empty fields.')

#SECTION 2 PAGE: GENERAL CONCLUSIONS

st.markdown('## General conclusions:')

st.markdown('1. The quality of data allows you to build social portraits of customers in the future.')
st.markdown('2. Demographic data of clients allows you to explore the administrative units in which the bank is more popular.')
st.markdown('3. Fields with data on the working status of the client are filled in incorrectly in some cases.')
st.markdown('4. When building a model, you will probably have to use data imbalance correction methods.')
st.markdown('5. Rare categories can lead to bad generalizing ability of the model.')



#SECTION 2 PAGE: MODEL TESTING

model = load_model('linear_model.pickle')
X_train = np.load('model_files/train_X_data.npy') 
X_test = np.load('model_files/test_X_data.npy')

X_test_raw = load_data('model_files/X_test.csv').reset_index()
columns = X_test_raw.columns

y_train = np.load('model_files/train_y_data.npy')
y_test = np.load('model_files/test_y_data.npy')


st.markdown('# Model testing')
st.markdown('## Conclusions about the model')
st.markdown('Logistic regression was chosen as the basic model for predicting the probability of clients responses. Reasons for choosing logistic regression are:')
st.markdown('- quick of learning and prediction;')
st.markdown('- possibility of soft classification;')
st.markdown('- mathematical interpretability of the probabilities')

st.markdown('The training of models showed that anomalies in the length of service (thay were found during EDA) do not affect the quality of the model. This allows us to leave this data in the training dataset without the need to correct potentially erroneous observations (**however, it is important to clarify the nature of the error**).')
#
st.sidebar.header('Testing response prediction for test clients')
selected_clf_treshold = st.sidebar.slider('Select treshold to predict positive (1) class', min_value=0.01, max_value=1.0, value=0.5, step=0.01)
selected_client = st.sidebar.selectbox('Index', (range(0, len(X_test))))
clf_report, weighted_metrics = get_classification_report_to_custom_threshold(model, X_test, y_test, selected_clf_treshold)

selected_single_pred_treshold = st.sidebar.slider('Select treshold to predict class for the selected sample', min_value=0.01, max_value=1.0, value=0.5, step=0.01)

prediction = get_prediction_by_index(model, X_test_raw, index=selected_client, treshold=selected_single_pred_treshold)

st.write(clf_report)
st.write(weighted_metrics)

st.markdown('## Seleceted client')
st.write(pd.DataFrame(X_test_raw.iloc[selected_client, :]).transpose())

st.write('Model prediction for the selected client is', prediction)

with st.spinner('Calculating features importances...'):
    exp = get_shap_values_by_index(model, X_train, X_test, columns, index=selected_client, treshold=selected_single_pred_treshold)
    time.sleep(1)
st.success('Done!')

fig, ax = plt.subplots()

ax = shap.plots.waterfall(exp[selected_client])

st.pyplot(fig) 


