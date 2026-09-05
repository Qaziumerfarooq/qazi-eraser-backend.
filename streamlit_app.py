import streamlit as st
import subprocess
import time
import requests

st.title("Qazi Eraser Backend")
st.write("Server is starting...")

# Start the server.py in the background
if 'server_process' not in st.session_state:
    proc = subprocess.Popen(["python", "server.py"])
    st.session_state['server_process'] = proc
    st.success("Server process started!")

# Show status
st.info("The eraser backend is running in the background.")

# Health check
try:
    res = requests.get("http://localhost:8317/health")
    if res.status_code == 200:
        st.success("Server is HEALTHY and Ready!")
except:
    st.warning("Server is still booting up...")
