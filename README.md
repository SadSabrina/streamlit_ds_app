# 👑 LOAN DATA: collecting from database, merging, analysis, model building...
### ... and deploying a data app using streamlit.
![Financial-Tips.png](https://ltdfoto.ru/images/2023/08/27/Financial-Tips.png)
## Сollecting from database

coming soon

## Merging

The data was obtained in the form of 9 csv tables.

- D_loan: information about 21126 loans from 15223 clients.
- D_clients: information about social features of 16000 clients.
- D_last_credit: information abot loan details related to 15223 clients.
- D_pens: dataset with for encoding retirement status. (0 — retirement=True, 1 - retirement=False)
- D_salary: dataset with family and personal income of 15223 clients.
- D_target: dataset with target feature to 15223 clients.
- D_work: dataset with for encoding work status. (0 — works=True, 1 - works=False, 2 — works=None)
- D_job: information about work of 15223 clients, contains NANs
- D_close_loan: infromation about loan closing status.

The data was combined in 7 steps, with the preservation of all samples with the target variable. Obtaining details you can see in the first section in the file `EDA.ipynb`

## Analysis

During EDA , it was analyzed

- data quality
- the relationship of continuous features with the target variable
- correctness of values in categorical features
- frequency of repeated loans
- balance in the values of the target variable

The main conclusions are presented in the form of a web page: [Exploratory analysis of bank loan data.](https://appdsappg-8q3vz3riyoqfpg8n5z5k4e.streamlit.app/)

## Model building

coming soon

## App details

coming soon

