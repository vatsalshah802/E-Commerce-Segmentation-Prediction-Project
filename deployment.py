import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Custom page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🔍",
    layout="wide",
)

# Load the model and metadata
try:
    model_data = joblib.load("Support_Vector_Machine.joblib")
    model = model_data["model"]  # Extract trained model
    model_features = model_data["features"]  # Extract feature names
except Exception as e:
    st.error("❌ Error loading the model! Please ensure the file 'Support_Vector_Machine.joblib' is present and correctly formatted.")
    st.stop()

# Main Title and Introduction
st.title("🔍 Customer Segmentation Dashboard")
st.markdown("""
Welcome to the **Customer Segmentation Dashboard**!  
Use the trained **SVM model** to predict customer **Lifetime Value (LTV) Clusters** and analyze customer data through interactive visualizations.
""")

# Sidebar for input features
st.sidebar.header("✨ **Input Customer Features**")
st.sidebar.markdown("Please provide the customer data below:")

# Sidebar input collection
input_data = {}
input_data['Recency'] = st.sidebar.number_input("📅 Recency (days since last purchase)", min_value=0, value=30)
input_data['Frequency'] = st.sidebar.number_input("🛍️ Frequency (number of purchases)", min_value=0, value=10)
input_data['Revenue'] = st.sidebar.number_input("💰 Revenue (total spending)", min_value=0, value=500)
input_data['RecencyCluster'] = st.sidebar.slider("📊 Recency Cluster (0-5)", min_value=0, max_value=5, value=2)
input_data['FrequencyCluster'] = st.sidebar.slider("📊 Frequency Cluster (0-5)", min_value=0, max_value=5, value=2)
input_data['RevenueCluster'] = st.sidebar.slider("📊 Revenue Cluster (0-5)", min_value=0, max_value=5, value=2)

# Dynamically calculate OverallScore
input_data['OverallScore'] = input_data['RecencyCluster'] + input_data['FrequencyCluster'] + input_data['RevenueCluster']

# Segment inputs as dropdowns
input_data['Segment_High-Value'] = st.sidebar.selectbox("💎 High-Value Segment (0/1)", [0, 1])
input_data['Segment_Low-Value'] = st.sidebar.selectbox("💼 Low-Value Segment (0/1)", [0, 1])
input_data['Segment_Mid-Value'] = st.sidebar.selectbox("📉 Mid-Value Segment (0/1)", [0, 1])

# Create DataFrame
input_df = pd.DataFrame([input_data])

# Ensure columns match model features
try:
    input_df = input_df[model_features]
except KeyError as e:
    st.error(f"❌ Input features do not match the model features. Missing columns: {e}")
    st.stop()

# Display User Input
st.markdown("### 📝 **Input Data Summary**")
st.dataframe(input_df.style.highlight_max(axis=0))

# Prediction Section
st.markdown("### 🔮 **LTV Cluster Prediction**")
if st.button("✨ Predict"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ The predicted LTV Cluster is: **{prediction[0]}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# Visualization Section
st.markdown("### 📊 **Visualize Customer Clusters**")
uploaded_file = st.file_uploader("📂 Upload a dataset for cluster visualization (CSV format)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.markdown("#### 📁 **Uploaded Dataset Preview**")
        st.dataframe(df.head())

        # Cluster distribution plot
        if "LTVCluster" in df.columns:
            st.markdown("#### 🎯 **LTV Cluster Distribution**")
            plt.figure(figsize=(12, 6))
            sns.countplot(data=df, x="LTVCluster", palette="coolwarm")
            plt.title("LTV Cluster Distribution", fontsize=16)
            plt.xlabel("LTV Cluster", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            st.pyplot(plt.gcf())
        else:
            st.warning("⚠️ The dataset does not contain an 'LTVCluster' column.")
    except Exception as e:
        st.error(f"❌ Error loading the dataset: {e}")

# Footer
st.markdown("---")
st.markdown("**Developed by Vatsal Shah | 🚀 Powered by Streamlit**")
