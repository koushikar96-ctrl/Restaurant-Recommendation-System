# utils.py
import pandas as pd
import streamlit as st
import base64
import os

def load_data(path="data/restaurants_sample.csv"):
    df = pd.read_csv(path)
    # sanitize column names
    df.columns = [c.strip() for c in df.columns]
    return df

def style_background():
    # If user placed an image at assets/bg.jpg, use it as background
    bg_path = "assets/bg.jpg"
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/jpg;base64,{b64}");
                    background-size: cover;
                    background-attachment: fixed;
                }}
                </style>
                """, unsafe_allow_html=True)
