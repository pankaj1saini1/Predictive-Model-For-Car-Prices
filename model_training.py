import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df2=pd.read_csv('E:/car2.csv')
le = LabelEncoder()
a= ['Brand','model','Transmission','Owner','FuelType']
for col in a:
    df2[col] = le.fit_transform(df2[col])
    label_mapping.append(dict(zip(le.classes_, range(len(le.classes_)))))
    # print(le.classes_,range(len(le.classes_)))
print(label_mapping[1])

# print(df2.head())

y = df2['AskPrice']
X = df2.drop(columns=['AskPrice'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# mo=LinearRegression()
mo = RandomForestRegressor(n_estimators=100, random_state=42)
mo.fit(X_train,y_train)


📌 Explanation of the Script
  . Loads the dataset
  . Encodes categorical variables
  . Splits data into training & test sets
  . Scales numerical features
  . Trains a RandomForest model
  . Saves the trained model for future use
