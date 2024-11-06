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
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![Screenshot 2024-10-19 185259](https://github.com/user-attachments/assets/5b33a5fb-6d60-4d71-a9c5-b63791a24998)

```
data.isnull().sum()
```
![Screenshot 2024-10-19 185307](https://github.com/user-attachments/assets/e358e1bd-dc8b-4673-ba09-5fceb21c9408)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![Screenshot 2024-10-19 185324](https://github.com/user-attachments/assets/4ffadee5-6d66-4f18-a789-877fb2f82bf4)

```
data2=data.dropna(axis=0)
data2
```
![Screenshot 2024-10-19 185334](https://github.com/user-attachments/assets/434f6949-e8c2-40b7-8745-ffa3fc9460eb)

```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![Screenshot 2024-10-19 185344](https://github.com/user-attachments/assets/78cf12a4-80fa-460c-a88d-7021fcd3243b)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![Screenshot 2024-10-19 185352](https://github.com/user-attachments/assets/4b7d6445-1a9a-4174-b3a4-ac2ddaac92ce)

```
data2
```
![Screenshot 2024-10-19 185402](https://github.com/user-attachments/assets/8647dca9-1ceb-40cb-988f-50477911a5d4)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![Screenshot 2024-10-19 185454](https://github.com/user-attachments/assets/f4a0de3f-ee83-4c35-a8a1-7a16432509fd)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-10-19 185509](https://github.com/user-attachments/assets/aef4955b-ef56-41a3-862e-5f206d3e2fef)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-10-19 185517](https://github.com/user-attachments/assets/b2daea74-d7b3-40d7-a42d-b0b68c143888)

```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-10-19 185529](https://github.com/user-attachments/assets/6d7ce2a3-b181-4844-8543-f37176a71475)

```
x=new_data[features].values
print(x)
```
![Screenshot 2024-10-19 185602](https://github.com/user-attachments/assets/637bc667-0a1e-4570-8084-6ce8852b6cef)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![Screenshot 2024-10-19 190159](https://github.com/user-attachments/assets/e6f750ed-7896-45b5-8386-ae6451298f46)

```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![Screenshot 2024-10-19 190204](https://github.com/user-attachments/assets/fb797c67-01c9-42d8-a5be-6a1ea2a74ea7)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![Screenshot 2024-10-19 190210](https://github.com/user-attachments/assets/508cbdaa-f682-4755-b3b7-683b3223712f)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![Screenshot 2024-10-19 190216](https://github.com/user-attachments/assets/abb3d85d-61d8-472e-89f3-d64966c9aa08)

```
data.shape
```
![Screenshot 2024-10-19 190220](https://github.com/user-attachments/assets/95d9b09c-2a72-42ae-8812-88dfa2f3c840)

```
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
![Screenshot 2024-10-19 190231](https://github.com/user-attachments/assets/0ed6934c-7a16-4570-9749-a990be3c4889)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![Screenshot 2024-10-19 190238](https://github.com/user-attachments/assets/a5a075e8-6588-48a7-93f4-f1215f18d8a7)

```
tips.time.unique()
```
![Screenshot 2024-10-19 190245](https://github.com/user-attachments/assets/da5c3eb4-47b2-43a1-88ec-d632af732fbf)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-10-19 190251](https://github.com/user-attachments/assets/a70a1d33-8aba-4396-81ab-1cc0474a97cf)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![Screenshot 2024-10-19 190256](https://github.com/user-attachments/assets/37a70b05-b258-49cf-8a47-4899bd53f0ca)



# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
