#KartikBhaba_1178819
import pandas as pd
import numpy as np
#Visualize the data
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#pre-process
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer,KNNImputer
#iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 
#to train the model in ML
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score 
#for classification tasks
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier 
#metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import joblib


# Dataset Load
file_path = "/Users/kartikbhaba/Desktop/WebHealthProject/heart_disease_uci.csv"  # Path to the uploaded dataset
df = pd.read_csv(file_path)

#Define catagorical colums
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(cat_cols)

#find only numeric columns
num_cols = df.select_dtypes(exclude='object').columns.tolist()
print(num_cols)

#find only boolean columns
bool_cols = df.select_dtypes(include='bool').columns.tolist()
print(bool_cols)
#Empty colums check
print((df.isnull().sum()/len(df) * 100).sort_values(ascending=False))


##impute missing values using iterative imputer (What does this mean- define and talk about in presentation)
imputer = IterativeImputer(max_iter=10,random_state=42)
#fit the imputer on trestbps
imputer.fit(df[['trestbps']])
#transform 
df['trestbps'] = imputer.transform(df[['trestbps']])
#Check remaining empty values 
print(df['trestbps'].isnull().sum())


imputer_1 = IterativeImputer(max_iter=10,random_state=42)
#fit_transform the imputer
df[['ca','oldpeak','thalch','chol']] = imputer_1.fit_transform(df[['ca','oldpeak','thalch','chol']])
#Checking imputer success
print((df.isnull().sum()/len(df) * 100).sort_values(ascending=False))


#Imputing rest of the values, that slope fbs exang restecg 

#imputing thal with rfc (Very impportant talk abount in presentation)
#define the function
def impute_missing_values_with_rf(df, column_name):
    """
    Impute missing values in a categorical column using RandomForestClassifier.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with missing values.
        column_name (str): The name of the categorical column to impute.
    
    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
#dataf copt 
    df = df.copy()
    #missing vals check 
    if df[column_name].isnull().sum() == 0:
        print(f"No missing values in column '{column_name}' to impute.")
        return df
    
    print(f"Imputing missing values in column '{column_name}' with RandomForestClassifier...")
    
    # Lab enco target col
    le = LabelEncoder()
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].apply(lambda x: np.nan if x.lower() == 'nan' else x)
    
    # enco NAN vals
    non_null_data = df[df[column_name].notnull()]
    null_data = df[df[column_name].isnull()]
    
    if null_data.empty:
        print(f"No missing values in '{column_name}' after preprocessing.")
        return df
    
    # enco target var
    non_null_data[column_name] = le.fit_transform(non_null_data[column_name].astype(str))

 #x-features, Y- target define
    X = non_null_data.drop(columns=[column_name])
    y = non_null_data[column_name]

    # 1-ht enco feats
    X_encoded = pd.get_dummies(X, drop_first=True)

    # feature null set align
    null_data_encoded = pd.get_dummies(null_data.drop(columns=[column_name]), drop_first=True)
    null_data_encoded = null_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Data avail check
    if null_data_encoded.empty:
        print("No matching features found between training data and null data for prediction.")
        return df

    # Train RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_model.fit(X_encoded, y)
    
#Pred miss vals
    predicted_values = rf_model.predict(null_data_encoded)

#Deco_pred vals to org vals
    decoded_values = le.inverse_transform(predicted_values)

# impute miss vals
    df.loc[df[column_name].isnull(), column_name] = decoded_values

    print(f"Missing values in '{column_name}' have been imputed successfully.")
    return df
df = impute_missing_values_with_rf(df, 'thal')

df = impute_missing_values_with_rf(df, 'slope')

df = impute_missing_values_with_rf(df, 'exang')

df = impute_missing_values_with_rf(df, 'restecg')

df = impute_missing_values_with_rf(df, 'fbs')
#Check err log

#checkingfor imputation
print(df.info())
df = df[df['trestbps'] != 0]
#resting heartrate outlier remove, talk about in pres
df = df[(df['trestbps'] <= 175) & (df['trestbps'] > 60)]
#outliers in thalch- maximum heart rate achieved) (Talk about in pres)
df = df[(df['thalch'] >= 70) & (df['thalch'] <= 235)]

#outliers in oldpeak - ST depression induced by exercise (pres)
#df = df[(df['oldpeak'] >= -2.55) & (df['oldpeak'] <= 4.25)]
df[df['oldpeak']<-2.55]
df[df['oldpeak']>4.25]

df = df[(df['chol'] > 30) & (df['chol'] < 500)]

df.info()













#mlrprocess (split and train)
X = df.drop(columns='id', axis=1)
y = df['num']

#Enco catag cols
columns_to_encode = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Lab-enco catagorical cols
for col in columns_to_encode:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Enco column

# x- drop y - targer
X = df.drop(columns=['id', 'num'], axis=1)  # Exclude 'id' and the target column 'num'
y = df['num']

# Train test split (35,65)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

















# used models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
}

#Mod Train
results = []
for model_name, model in models.items():
    print(f"Training {model_name}...")
    try:
        # Train
        model.fit(X_train, y_train)
        
        # Test set predict
        y_pred = model.predict(X_test)
        
        # accuracy score disp
        acc = accuracy_score(y_test, y_pred)
        results.append((model_name, acc))
        
        # result disp
        print(f"{model_name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print(f"Error training {model_name}: {e}")

# Best mod out put 
if results:
    best_model_name, best_model_score = max(results, key=lambda x: x[1])
    print("\nBest Model:")
    print(f"{best_model_name} with accuracy: {best_model_score:.4f}")
else:
    print("No valid model could be trained.")


print(df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False), '\n')

missing_data_cols = df.isnull().sum()[df.isnull().sum()>0].index.tolist()

print ('List of columns with empty or missing data: ',missing_data_cols, '\n')

# Non-Num cols
non_num_cols = df.select_dtypes(include='object').columns.tolist()
print('List of non-numerical columns: ', non_num_cols, '\n')

#Numerical Cols
num_cols = df.select_dtypes(exclude='object').columns.tolist()
print('List of numerical columns: ', num_cols, '\n')

# Boolean Cols
bool_cols = ['fbs', 'exang']
print('List of boolean columns: ', bool_cols)

print(df.info())

# Save the best model
if results:
    best_model_name, best_model_score = max(results, key=lambda x: x[1])
    best_model = models[best_model_name]  # Get the best model object
    joblib.dump(best_model, "heart_disease_model.pkl")
    print(f"Best model ({best_model_name}) saved as 'heart_disease_model.pkl'")
else:
    print("No valid model could be trained.")


#Fix this 