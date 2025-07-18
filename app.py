import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# === Load model & scaler ===
model_path = Path(__file__).parent / "notebooks" / "sales_prediction.pkl"
scaler_path = Path(__file__).parent / "notebooks" / "scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("🛒 BigMart Sales Prediction")

# === Input Fields ===
item_weight = st.number_input("Item Weight (kg)", 1.0, 50.0, 10.0)
item_visibility = st.number_input("Item Visibility", 0.0, 0.3, 0.05)
item_mrp = st.number_input("Item MRP (₹)", 10.0, 300.0, 100.0)
outlet_year = st.number_input("Outlet Establishment Year", 1980, 2025, 2004)

item_fat = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
item_type = st.selectbox("Item Type", [
    "Baking Goods", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegetables",
    "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others",
    "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"
])
outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
outlet_location = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
new_item_type = st.selectbox("New Item Type", ["Food", "Non-Consumable", "Drinks"])

# === Encoding Maps ===
fat_map = {"Low Fat": 0, "Regular": 1}
item_type_map = {
    "Baking Goods": 0, "Canned": 1, "Dairy": 2, "Frozen Foods": 3,
    "Fruits and Vegetables": 4, "Hard Drinks": 5, "Health and Hygiene": 6,
    "Household": 7, "Meat": 8, "Others": 9, "Seafood": 10,
    "Snack Foods": 11, "Soft Drinks": 12, "Starchy Foods": 13
}
size_map = {"Small": 2, "Medium": 1, "High": 0}
location_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
outlet_map = {
    "Grocery Store": 0, "Supermarket Type1": 1,
    "Supermarket Type2": 2, "Supermarket Type3": 3
}
new_type_map = {"Food": 0, "Non-Consumable": 1, "Drinks": 2}

# === Construct Input DataFrame (exact order) ===
input_data = pd.DataFrame([[
    item_weight,
    fat_map[item_fat],
    item_visibility,
    item_type_map[item_type],
    item_mrp,
    outlet_year,
    size_map[outlet_size],
    location_map[outlet_location],
    outlet_map[outlet_type],
    new_type_map[new_item_type],
    2025 - outlet_year
]], columns=[
    "Item_Weight",
    "Item_Fat_Content",
    "Item_Visibility",
    "Item_Type",
    "Item_MRP",
    "Outlet_Establishment_Year",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
    "New_Item_Type",
    "Outlet_Years"
])

# === Scale numerical columns ===
num_cols = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Years"]
input_data[num_cols] = scaler.transform(input_data[num_cols])
correct_order = [
    'Item_Weight',
    'Item_Fat_Content',
    'Item_Visibility',
    'Item_Type',
    'Item_MRP',
    'Outlet_Establishment_Year',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type',
    'New_Item_Type',
    'Outlet_Years'
]
input_data = input_data[correct_order]

# === Preview & Predict ===
st.subheader("🔍 Input Preview")
st.dataframe(input_data)

if st.button("Predict Sales"):
    try:
        pred = model.predict(input_data)
        st.success(f"✅ Predicted Sales: ₹{float(pred):.2f}")

    except Exception as e:
        st.error("❌ Prediction Failed")
        st.exception(e)
