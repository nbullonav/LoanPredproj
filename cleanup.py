import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
# Load libraries for data clean up
import missingno as mno
import random
import numpy as np

df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()
df.isna().sum()/4269

#MAPPING
#variable education
df['education'] = df['education'].str.strip()
df['education'].value_counts()
df['education_encoded'] = 0
df.loc[df['education']=='Graduate', 'education_encoded'] = 1
df.loc[df['education']=='Not Graduate', 'education_encoded'] = 0
df['education_encoded'].value_counts()
#variable: self_employed
df['self_employed'] = df['self_employed'].str.strip()
df['self_employed_encoded'] = 0
df.loc[df['self_employed']=='Yes', 'self_employed_encoded'] = 1
df.loc[df['self_employed']=='No', 'self_employed_encoded'] = 0
#variable: Cibil score
df['cibil_score_encoded'] = 0
df.loc[ df['cibil_score']<300,'cibil_score_encoded'] = 0
df.loc[(df['cibil_score']>=300) & (df['cibil_score']<=549),'cibil_score_encoded'] = 1 # poor
df.loc[(df['cibil_score']>=550) & (df['cibil_score']<=649),'cibil_score_encoded'] = 2 # fair
df.loc[(df['cibil_score']>=650) & (df['cibil_score']<=749),'cibil_score_encoded'] = 3 # good
df.loc[(df['cibil_score']>=750) & (df['cibil_score']<=900),'cibil_score_encoded'] = 4 # excelent
#Loan Status
df['loan_status'] = df['loan_status'].str.strip()
df['loan_status_encoded'] = 0
df.loc[df['loan_status']=="Rejected",'loan_status_encoded'] = 0 #rejected
df.loc[df['loan_status']=="Approved",'loan_status_encoded'] = 1 #approved

plt.rcParams['figure.figsize']=(12,10)
df[['no_of_dependents','education_encoded','self_employed_encoded','income_annum','loan_amount','loan_term','cibil_score_encoded','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']].hist(bins=50)
#plt.show()
plt.savefig("hist.jpg")

# correlation graph
df[['no_of_dependents','education_encoded','self_employed_encoded','income_annum','loan_amount','loan_term','cibil_score_encoded','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value']].corr()

# pairplot graph
plt.rcParams['figure.figsize']=(10,8)
sb.pairplot(df[['no_of_dependents','education_encoded','self_employed_encoded','income_annum','loan_amount','loan_term','cibil_score_encoded','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value','loan_status_encoded']],hue = 'loan_status_encoded')
#plt.show()
plt.savefig("pairplot.jpg")

# correlation graph
plt.figure(figsize = (10,10))
sb.heatmap(df[['no_of_dependents','education_encoded','self_employed_encoded','income_annum','loan_amount','loan_term','cibil_score_encoded','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value','loan_status_encoded']].astype(float).corr(),square = True, annot = True, vmax = 1, linewidths = 0.1)
#plt.show() 
plt.savefig("correlation.jpg")

# download data to load into MS Azure
df_download = df[['no_of_dependents','education_encoded', 'self_employed_encoded','income_annum','loan_amount','loan_term','cibil_score_encoded','residential_assets_value','commercial_assets_value','luxury_assets_value','bank_asset_value','loan_status_encoded']]
df_download.to_csv("loandata_clean.csv", index = False)