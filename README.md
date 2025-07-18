#  Stores_Sales_Prediction
# 🛒 BigMart Sales Prediction App

This is a machine learning web application that predicts sales for BigMart products based on various product and outlet features. The app is built using **Streamlit** and leverages a trained **XGBoost regression model** to estimate the sales for a given product.

---

## 📌 Features

- 📈 Predicts sales (`Item_Outlet_Sales`) for products in BigMart
- 🧠 Uses an XGBoost Regression Model trained on historical data
- ⚙️ Automatic feature scaling using StandardScaler
- 🌐 Interactive and easy-to-use web interface built with Streamlit
- 🔍 Inputs include item details like MRP, type, outlet location, and more

---

## 🚀 Tech Stack

- Python 3.10+
- Streamlit
- Pandas
- Scikit-learn
- XGBoost
- Joblib (for saving/loading model and scaler)

---

## 🧪 How the Model Was Built

The model was trained on a dataset with the following features:

- `Item_Weight`
- `Item_Fat_Content`
- `Item_Visibility`
- `Item_Type`
- `Item_MRP`
- `Outlet_Establishment_Year`
- `Outlet_Size`
- `Outlet_Location_Type`
- `Outlet_Type`
- `New_Item_Type` (derived)
- `Outlet_Years` (derived)

Label encoding and scaling were performed before training the XGBoostRegressor.

---
