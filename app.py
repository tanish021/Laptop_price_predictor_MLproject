import streamlit as st
import pickle
import numpy as np
import pandas as pd # Ensure pandas is imported

# Set Page Config
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

st.title("Laptop Price Predictor")

# Load the model and the dataframe
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# --- Input Fields ---
company = st.selectbox('Brand', df['Company'].unique())
type_name = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop (kg)', value=1.5)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])
screen_size = st.number_input('Screen Size (in inches)', value=15.6)

resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440',
    '2304x1440'
])

cpu = st.selectbox('CPU', df['Cpu_brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

# --- Prediction Logic ---
# --- Prediction Logic ---
if st.button('Predict Price'):

    # 1. Convert categorical "Yes/No" to 1/0 FIRST
    ts_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # 2. Calculate PPI
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5 / screen_size

    # 3. NOW create the DataFrame (ts_val is now defined)
    query_df = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [ts_val],
        'IPS': [ips_val],
        'ppi': [ppi],
        'Cpu_brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu_brand': [gpu],
        'os': [os]
    })

    # 4. Make Prediction
    prediction = pipe.predict(query_df)
    st.title(f"The predicted price is ₹{int(np.exp(prediction[0]))}")