import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

GENERAL_DF = 'datasets/general_df.csv'
GENERAL_DF_CLEANED = 'datasets/general_df_cleaned.csv'
LOANS_DF = 'datasets/loans_df.csv'

clients_without_loans_now = 9477
clients_with_loans_now = len(general_df) - clients_without_loans_now

continuous_features = ['credit', 'fst_payment', 'age', 'child_total', 'dependants',
                       'own_auto', 'personal_income', 'work_time', 'closed_loans_count',
                       'total_loans_count']

#SECTION 1 DATA LOADING
def load_data(path: str):
    return pd.read_csv(path, index_col=0)

#pie chart function
def get_rare_pie(data: pd.Series, limits_count: int = 5):

    """Function to draw pie chart with rare categories"""
    limits_dict = {}

    for i in range(1, limits_count + 1):
        limits_dict[i] = int(len(data) / 100 * i + 1)

    rare_values = {}
    data_to_pie_dict_categories = {}
    data_to_pie = None

    for i in range(1, limits_count + 1):

        if i == 1:
            rare_values[i] = data.value_counts()[data.value_counts() < limits_dict[i]].index

        else:
            rare_values[i] = data.value_counts()[
                (data.value_counts() < limits_dict[i]) & (data.value_counts() >= limits_dict[i - 1])].index

        data_to_pie_dict_categories[
            f'Categories ({len(rare_values[i])}) in which there are less samples than {i}% of data'] = len(
            rare_values[i])

    pie_labels_categories = data_to_pie_dict_categories.keys()
    pie_values_categories = data_to_pie_dict_categories.values()

    fig = plt.figure(figsize=(6, 6))

    plt.pie(pie_values_categories, autopct='%1.0f%%',
            colors=sns.color_palette('Set2'), shadow=True,
            wedgeprops={"edgecolor": "white",
                        'linewidth': 2,
                        'antialiased': True})
    plt.legend(pie_labels_categories, bbox_to_anchor=(1, 0.7))
    plt.title('Categories in which samples are less than % of the data', weight='bold')
    plt.show()

    return fig

general_df = load_data(GENERAL_DF)
general_df_cleaned = load_data(GENERAL_DF_CLEANED)
loans_df = load_data(LOANS_DF)

#SECTION 2 EDA PAGE: TITLE AND DESCRIPTION
st.title('Exploratory analysis of bank loan data.')
st.divider()
st.text('Report on the preliminary analysis of data on customer loans. \n'
        'Data has been merged from multiple tables in a database. \n'
        'Here are the results of the analysis of the quality, \n'
        'completeness of the data and further opportunities to work on \n'
        'the basis of these data.')

#SECTION 3 EDA PAGE: EDA BASIC CONCLUSIONS
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

#SECTION 3 EDA PAGE: EDA CONCLUSIONS ABOUT TARGET AND FEATURES QUALITY

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

#SECTION 4 EDA PAGE: GENERAL CONCLUSIONS

st.markdown('## General conclusions:')

st.markdown('1. The quality of data allows you to build social portraits of customers in the future.')
st.markdown('2. Demographic data of clients allows you to explore the administrative units in which the bank is more popular.')
st.markdown('3. Fields with data on the working status of the client are filled in incorrectly in some cases.')
st.markdown('4. When building a model, you will probably have to use data imbalance correction methods.')
st.markdown('5. Rare categories can lead to bad generalizing ability of the model.')
