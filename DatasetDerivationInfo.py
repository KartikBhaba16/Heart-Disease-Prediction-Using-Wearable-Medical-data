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



# Load the dataset
file_path = "/Users/kartikbhaba/Desktop/WebHealthProject/heart_disease_uci.csv" 
df = pd.read_csv(file_path)
df.info()
#describe the age (Talk about in presentation )
df['age'].describe()

#Plot age distribution, (Talk in presentation)
sns.histplot(data=df, x='age', kde=True,color='green')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.show()

print("The mean of age is: ", df['age'].mean())
print("The median of age is: ", df['age'].median())
print("The mean of age is: ",df['age'].mode()[0])
print("The max age is: ", df['age'].max())
print("The min age is: ", df['age'].min())
#males female age - analysis (Talk about in presentation)
print(df['sex'].value_counts())

#derived from the above value count 
male_count = 726
female_count = 194
total_count = male_count + female_count
#total count=920
#Talk in presentation about data male count etc
male_percentage = (male_count/total_count) * 100
print("The percentage of male's value count in data is:", round(male_percentage,2),'%')
female_percentage = (female_count/total_count) * 100
print("The percentage of female's value count in data is:", round(female_percentage,2),'%')
#gender percentage for gender split (Talk in presentation)
sns.barplot(x=['Male', 'Female'], y=[male_percentage, female_percentage])
plt.title('Percentage by Gender')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.show()

#Chest pain 0 split by gender, types of cp and descriptives (Talk about in presentation)
print(df['cp'].value_counts())
df.groupby('cp')['age'].value_counts().sort_values(ascending=False)
#chest pain type split graph (talk about in presentation)
sns.countplot(df, x='cp', hue='sex')
plt.title('Chest pain by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

#Resting bloop pressure descriptives and splits (trestbps)
print('\n Resting Blood Pressure - Stats \n')
print(df['trestbps'].describe())
sns.histplot(data=df, x='trestbps',kde=True)
plt.title('Resting blood pressure count split')
plt.xlabel('Resting Blood Pressure (tresrbps)')
plt.ylabel('Count')
plt.show()
#missing value check in trestbps
print('\n null values for trestbps')
print(df['trestbps'].isnull().sum()) #null values confirmed 
#group by gender (Talk about in presentation)
print(df.groupby('sex')['trestbps'].value_counts())


#Cholestrol - cholestrol measure (talk about in presentation)
print(df['chol'].value_counts())
print('\n null values for chol: ')
print(df['chol'].isnull().sum())

#Fasting blood sugar (talk about in presentation)
print(df['fbs'])
print('\n null values for fbs: ')
print(df['fbs'].isnull().sum())

#RestingECG-restecg - ecg obs at resting condition (Talk about in presentation)
print(df['restecg'].value_counts())
print('\nnull values for restecg:')
print(df['restecg'].isnull().sum())
sns.countplot(data=df, x='restecg', hue='sex')
plt.title('Resting ECG')
plt.ylabel('Count')
plt.show()

#Thalch (Maximum heart rate achieved) - Talk about in presentation
print(df['thalch'])
print(df['thalch'].value_counts())
print('\nnull values for thalch:')
print(df['thalch'].isnull().sum())

#Exang stats - exercise induced angina (t,f)
print(df['exang'])
print(df['exang'].isnull().sum())


#OldPeak Stats - st depression induced by exercise relative to rest
print(df['oldpeak'])
print(df['oldpeak'].isnull().sum())

#Slope - the slope of the peak exercise st segment
print(df['slope'].value_counts())
print(df['slope'].isnull().sum())

#CA stats - number of major arteries colored by fluroscopy
print(df['ca'].value_counts().head(4))
print(df['ca'].isnull().sum())

#Thal- (Thalassamia type)
print(df.groupby('sex')['thal'].value_counts())
sns.countplot(data=df, x='thal', hue= 'sex')
plt.title('Thalassamia type split')
plt.xlabel('Thalassamia')
plt.ylabel('Count')
plt.show()

#Num - Target variable diagnosis /severity (Very Imp talk about in presentation)
print(df['num'].value_counts())
sns.countplot(data=df, x='num', palette='viridis')
plt.title('Distribution of Heart Disease Severity (num)')
plt.xlabel('Heart Disease Severity (num)')
plt.ylabel('Count')
plt.show()
#Avg ager as per severity of diagnosis for heart disease (Very imp talk about in presentation)
avg_age_per_num = df.groupby('num')['age'].mean()
print(avg_age_per_num)


#outlier check
sns.boxplot(df['chol'],color='blue')
plt.show()
sns.boxplot(df['oldpeak'],color='blue')
plt.show()
sns.boxplot(df['trestbps'],color='blue')
plt.show()
sns.boxplot(df['thalch'],color='blue')

print(df.info())