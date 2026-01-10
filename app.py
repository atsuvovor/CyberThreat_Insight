#Streamlit Backend Dashboard

import streamlit as st
import subprocess

st.title("🛡️ Cyber Threat Insight Platform")

stage = st.selectbox(
    "Select Pipeline Stage",
    ["all", "dev", "inference", "production", "attack", "dashboard"]
)

if st.button("Run Pipeline"):
    with st.spinner("Executing pipeline..."):
        subprocess.run(["python", "main.py", "--stage", stage])
    st.success("Execution completed!")
