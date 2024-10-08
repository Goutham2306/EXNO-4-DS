# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
      
```
NAME : GOUTHAM.K
REG NO : 212223110019
```
```python
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/ceca842c-3630-450b-8a5d-a828fe86ec49)


```python

data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/0855c83c-c104-45c6-83f1-7caf0c6bfded)


```python

missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/e36e28a0-3f58-4897-aa02-81cbb13e273e)

```python

data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/de74acb2-682d-4ba7-b831-f07d4725f7a7)

```python
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/828d78c4-1515-4358-a05b-b2fd0935340c)


```python
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/8400efe8-6588-4f6e-9596-d9fcce88caf3)

```python
data2
```
![image](https://github.com/user-attachments/assets/e11a30e1-cc49-4fed-98ba-125cf8647079)

```python
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/3c076249-c989-4936-a669-2dfe394a6ecb)

```python

columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/b1e2c012-5f8d-4ef8-bafc-fd7f5cfdff2a)

```python


features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/7af71129-92cb-4841-b80c-8f3ba652c77f)

```python
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/3862ae5b-bcfe-46e2-9d71-91c39dc6ec9a)

```python

x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/7270c6cc-54c2-4bf3-8903-4e0a07cf340f)

```python

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```

```python

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/a6eedfe3-aedd-4500-958f-6faafd54f464)
```python

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/0e56ff41-2f35-4d01-b479-53547391567b)
```python

print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/4af5ed3f-362a-40c6-a438-c89f31584e51)
```python

data.shape
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/1986f990-26e6-4b42-acfc-b2a6e52f8042)
```python

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/20777b0d-3cdb-4ae9-80e4-1f76ed093191)
```python

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/6d6f7ff2-b1da-4568-9cd1-cb6fa9553cd6)
```python

tips.time.unique()
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/f77bc757-8a31-4a5d-be15-5a447e6549c6)
```python

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/06365e9f-f51b-4cf6-ab04-8a136726a025)
```python

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/Yamunaasri/EXNO-4-DS/assets/115707860/6adc4da7-421c-458f-9ec6-f6158aa6f731)
# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
