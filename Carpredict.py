import  pandas as pd
pd.set_option("display.max_columns",15)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# d="Mozilla/5.0 (iPhone; CPU iPhone OS 17_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) GSA/343.0.695551749 Mobile/15E148 Safari/604.1"
# from bs4 import BeautifulSoup
# import requests
# header={"user-agent":d}
# b="https://www.olx.in/en-in/cars_c84?page="
# i=1
# price1=[]
# subt2=[]
# car=[]
# olc1=[]
# while i <14:
#     a=requests.get(b+str(i),headers=header)
#     # print(a.text)
#     e=BeautifulSoup(a.content,"html.parser")
#     hi=e.find_all("a","href")
#     for h in hi:
#         print(h.text)
    # prices = e.find_all('span', {'data-aut-id': 'itemPrice'})
    # for price in prices:
    #     price1.append(price.text)
    #
    # subtitle = e.find_all('div', {'data-aut-id': 'itemSubTitle'})
    # for subt in subtitle:
    #     subt2.append(subt.text)
    #     title = e.find_all('div', {'data-aut-id': 'itemTitle'})
    # for titl in title:
    #     car.append(titl.text)
    # loc=e.find_all("div",{"data-aut-id":"itemDetails"})
    # for l in loc:
    #     olc1.append(l.text)
    #     print("done")
    #
    # i+=1
# z=pd.DataFrame(price1,columns=["Price"])
# z["Model Year"]=subt2
# z['Car Name']=car
# z["Location"]=olc1
# z.to_csv("d:/114.csv")

# df=pd.read_csv('d:/114.csv')
# # print(df.head())
# # print(df.columns)
# df.drop(['Unnamed: 0'],axis=1,inplace=True)
#
# df['Model_year']=df['Model Year'].apply(lambda x: x.split('-')[0])
#
# df['KMS DRIVEN']=df['Model Year'].apply(lambda x: x.split('-')[-1])
# df.drop(["Model Year"],axis=1,inplace=True)
#
# df['Lo']=df['Location'].apply(lambda x: x.split(' ')[-1])
# print(df.head())


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor



df2=pd.read_csv('E:/car2.csv')
# print(df2.head())

# print(df2.columns)

# print(df2.isnull().sum())

df2.dropna(axis=0,inplace=True)
# print(df2.isnull().sum())

# print(df2.head())

df2['AskPrice']=df2['AskPrice'].apply(lambda x: x.split(' ')[-1].replace(',', '')).astype(int)

df2['kmDriven']=df2['kmDriven'].apply(lambda x: x.split(' ')[0].replace(',', '')).astype(float)
df2['kmDriven']=df2['kmDriven'].astype(int)

# print(df2.head())

# print(df2.info())

df2.drop(columns=['PostedDate', 'AdditionInfo','Age'],inplace=True)
print(df2.columns)
# print(df2.head())
label_mapping=[]
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
# pre=mo.predict(X_test)




brand_mapping=label_mapping[0]
model_mapping=label_mapping[1]
transmission_mapping=label_mapping[2]
owner_mapping=label_mapping[3]
fuel_mapping=label_mapping[4]


# New car details
# new_car = pd.DataFrame({
#     'Brand': [38],
#     'model': [198],
#     'Year': [2018],
#     'Transmission': ['Automatic'],
#     'Owner': ['First Owner'],
#     'FuelType': ['Petrol'],
#     'kmDriven': [50000]
# })
#
# new_car['Age'] = 2024 - new_car['Year']
# new_car.drop(columns=['Year'], inplace=True)
#
# # Encode categorical features safely
# for col in ['Brand', 'model', 'Transmission', 'Owner', 'FuelType']:
#     if new_car[col].values[0] in le.classes_:
#         new_car[col] = le.transform(new_car[col])
#     else:
#         new_car[col] = -1
#
# # Ensure correct column order
# new_car = new_car.reindex(columns=X.columns, fill_value=0)
#
# # Predict price
# predicted_price = mo.predict(new_car)
# print(f"Predicted Car Price: {predicted_price[0]}")


import streamlit as st
image_url="https://images.pexels.com/photos/7144212/pexels-photo-7144212.jpeg?auto=compress&cs=tinysrgb&w=600)"
st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
)

st.title("Predictive Model For Car Prices ")
brand = st.selectbox("Select Car Brand", list(brand_mapping.keys()))
model_name = st.selectbox("Select Car Model", list(model_mapping.keys()))
year = st.number_input("Manufacturing Year", min_value=2000, max_value=2025, step=1)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000)

transmission = st.selectbox("Select Transmission Type", list(transmission_mapping.keys()))
owner = st.selectbox("Select Ownership Type", list(owner_mapping.keys()))
fuel = st.selectbox("Select Fuel Type", list(fuel_mapping.keys()))

brand_encoded = brand_mapping[brand]
model_encoded = model_mapping[model_name]
transmission_encoded = transmission_mapping[transmission]
owner_encoded = owner_mapping[owner]
fuel_encoded = fuel_mapping[fuel]

input_data = pd.DataFrame([[brand_encoded, model_encoded,year, km_driven, transmission_encoded, owner_encoded, fuel_encoded]],
                          columns=['Brand', 'model', 'Year', 'kmDriven', 'Transmission','Owner','FuelType'])

# Predict Button
if st.button("Predict Price"):
    price = mo.predict(input_data)[0]
    st.success(f"Estimated Car Price: {price:,.2f}")