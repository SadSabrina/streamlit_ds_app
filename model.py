import pandas as pd
import pickle
import shap
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import classification_report


PATH = 'datasets/encoded_data.csv'

def load_data(path: str):
    return pd.read_csv(path, index_col=0)

def load_model(path='linear_model.pickle'):
    with open(path, 'rb') as f:
        model = pickle.load(f)

    return model


def get_train_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)
    sc = StandardScaler()
    X_train_transformed = sc.fit_transform(X_train)
    X_test_transformed = sc.transform(X_test)
    
    return X_train_transformed, X_test_transformed, y_train, y_test, X_test

def fit_and_save_linear_model(X_train, y_train, path_to_save='linear_model.pickle'):

    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    
    with open(path_to_save, 'wb') as f:
        pickle.dump(model, f)

    print(f'Model was saved to {path_to_save}')


def get_classification_report_to_custom_threshold(model, X_test_data, y_test_data, treshold=0.5):

    predictions_pr = model.predict_proba(X_test_data)[:, 1] 
    predictions_cl = model.predict_proba(X_test_data)[:, 1] > treshold

    #Weighted metrics
    
    recall = recall_score(y_test_data, predictions_cl, average='weighted', zero_division=0)
    precision = precision_score(y_test_data, predictions_cl, average='weighted', zero_division=0)
    f1 = f1_score(y_test_data, predictions_cl, average='weighted', zero_division=0)

    #Classification report

    cl_report = pd.DataFrame(classification_report(y_test_data, predictions_cl, output_dict=True)).transpose().drop(['macro avg', 'weighted avg', 'accuracy'])

    weighted_data = pd.DataFrame({'precision' : precision, 'recall' : recall, 'f1': f1}, index=['Weighted metrics'])

    return cl_report, weighted_data

def get_prediction_by_index(model, X_test_data, index=1, treshold=0.5):

    print()
    predictions = model.predict_proba(X_test_data)[:, 1] > treshold
    selected_idx = predictions[index]

    return int(selected_idx)
    
    

def get_shap_values_by_index(model, X_train_data, X_test_data, columns, index=1, treshold=0.5):

    predictions = model.predict_proba(X_test_data)[:, 1] > treshold
    
    f = lambda x: model.predict_proba(x)[:, 1]
    med = np.median(X_train_data, axis=0).reshape((1, X_train_data.shape[1]))
    explainer = shap.Explainer(f, med)
    
    shap_values = explainer(X_test_data)
    
    exp = shap.Explanation(shap_values, 
                  shap_values.base_values, 
                  data=X_train_data, 
                  feature_names=columns)

    return exp #shap.waterfall_plot(exp[index])

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)