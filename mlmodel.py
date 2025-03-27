import pandas as pd
import numpy as np
# ML libraries
import sklearn.model_selection as model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

#load clean data
df = pd.read_csv("loandata_cleaned.csv")

#VARIABLE CORRELATION ANALYSIS

# DATA BALANCING

# Train and Test
X = df[['income_annum','loan_amount','cibil_score_encoded','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']]
y = df['loan_status_encoded']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.80, random_state = 123 )

df_train = pd.concat([X_train,y_train],axis = 1)
df_train.head()


# Calculate qty of 0 & 1 within dataset
count_class_0, count_class_1 = df_train['loan_status_encoded'].value_counts()
# calculating datasets:
df_class_0 = df_train[df_train['loan_status_encoded']==0]
df_class_1 = df_train[df_train['loan_status_encoded']==1]

count_class_0, count_class_1, len(df_class_0), len(df_class_1)

# Undersampling
df_class_1.sample(2)
df_class_0_under = df_class_0.sample(count_class_1, random_state = 123)
df_train_u = pd.concat([df_class_0_under,df_class_1], axis=0)
df_train_u['loan_status_encoded'].value_counts()

# Oversampling
df_class_1_over = df_class_1.sample(count_class_0, random_state = 123, replace= True)
df_train_o = pd.concat([df_class_0,df_class_1_over], axis=0)
#df_train_o['loan_status_encoded'].value_counts()

## Data Evaluation and Confusion Matrix

# 0) Original data

# Call the model
model = DecisionTreeClassifier(random_state = 123)
# Train the model
model.fit(X_train,y_train)
# Generate the prediction for the test
y_pred = model.predict(X_test)
# Metrics eval
print(accuracy_score(y_test,y_pred))
# Confusion Matrix
print('Confusion Matrix: ')
print(confusion_matrix(y_test,y_pred))
# Classification Report
print('Classification Report: ')
print(classification_report(y_test,y_pred))

# 1) Data eval with undersampling

# Call the model
model = DecisionTreeClassifier(random_state = 123)
# Generación de X_train y y_train
X_train_u = df_train_u[['income_annum','loan_amount','cibil_score_encoded','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']]
y_train_u = df_train_u['loan_status_encoded']
# Train the model
model.fit(X_train_u,y_train_u)
# Generate the prediction for the test
y_pred = model.predict(X_test)
# Metrics eval
print(accuracy_score(y_test,y_pred))
# Confusion Matrix
print('Confusion Matrix: ')
print(confusion_matrix(y_test,y_pred))
# Classification Report
print('Classification Report: ')
print(classification_report(y_test,y_pred))

# Data Eval w/ Oversampling

# LLamar al modelo
model = DecisionTreeClassifier(random_state = 123)
# Generación de X_train y y_train
X_train_o = df_train_o[['income_annum','loan_amount','cibil_score_encoded','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']]
y_train_o = df_train_o['loan_status_encoded']
# Entrenar el modelo
model.fit(X_train_o,y_train_o)
# Generar la predicción para el test
y_pred = model.predict(X_test)
# evaluar metricas
print(accuracy_score(y_test,y_pred))
# matriz de confusión
print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))
# Reporte de Clasificación:
print('Classification Report: ')
print(classification_report(y_test,y_pred))

#Data Sampling Comparison

SamplingTable = pd.DataFrame({'Data Balance':['Original','Under','Over'],
                      'Accuracy': [0.98,0.73,0.72],
                      'Recall':[0,0.75,0.69],
                      'Precision':[0,0.05,0.05]})
SamplingTable.head()

## OTHER MODELS

# XGBOOST Algorithm (w/o Tuning)
# Call the model
model1 = xgb.XGBClassifier(random_state = 123)
# Train the model
model1.fit(X_train_u,y_train_u)
# Generate prediction for the test
y_pred1 = model1.predict(X_test)
# Metrics eval
print(accuracy_score(y_test,y_pred1))
# Confusion Matrix
print('Confusion Matrix: ')
print(confusion_matrix(y_test,y_pred1))
# Classification Report
print('Classification Report: ')
print(classification_report(y_test,y_pred1))

# XGBOOST Algorithm (with Tuning)
# Call the model
model2 = xgb.XGBClassifier(random_state = 123, n_estimators = 20, max_depth = 8, learning_rate=0.1, subsample = 0.5)
# Train the model
model2.fit(X_train_u,y_train_u)
# Generate prediction for the test
y_pred2 = model2.predict(X_test)
# Metrics eval
print(accuracy_score(y_test,y_pred2))
# Confusion Matrix
print('Confusion Matrix: ')
print(confusion_matrix(y_test,y_pred2))
# Classification Report
print('Classification Report: ')
print(classification_report(y_test,y_pred2))

# RANDOM FOREST Algorithm (w/o Tuning)
# Call the model
model3 = RandomForestClassifier(random_state = 123)
# Train the model
model3.fit(X_train_u,y_train_u)
# Generate prediction for the test
y_pred3 = model3.predict(X_test)
# Metrics eval
print(accuracy_score(y_test,y_pred3))
# Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred3))
# Classification Report
print('Classification Report')
print(classification_report(y_test,y_pred3))

# Algoritmo de RANDOMFOREST (w/ Tuning)
# call model
model4 = RandomForestClassifier(random_state = 123, n_estimators=20, max_depth=8)
# train model
model4.fit(X_train_u,y_train_u)
# generate test prediction
y_pred4 = model4.predict(X_test)
# evaluate metrics
print(accuracy_score(y_test,y_pred4))
# confusion matrix
print('Confusion Matrix: ')
print(confusion_matrix(y_test,y_pred4))
# Classification report:
print('Classification Report: ')
print(classification_report(y_test,y_pred4))


#Trying new prediction pronostics

## Change values to create a new prediction
datos_dummy = pd.DataFrame({'income_annum':[45000],
                            'loan_amount':[50000],
                            'cibil_score_encoded':[3],
                            'residential_assets_value':[140000],
                            'commercial_assets_value':[0],
                            'luxury_assets_value':[0],
                            'bank_asset_value':[10000]})
datos_dummy.head()

model4.predict(datos_dummy)

## Change values to create a new prediction
datos_dummy2 = pd.DataFrame({'income_annum':[30000],
                            'loan_amount':[30000],
                            'cibil_score_encoded':[1],
                            'residential_assets_value':[0],
                            'commercial_assets_value':[0],
                            'luxury_assets_value':[0],
                            'bank_asset_value':[15000]})
datos_dummy2.head()

model4.predict(datos_dummy2)