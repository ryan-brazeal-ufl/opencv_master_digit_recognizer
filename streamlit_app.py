import streamlit as st
import threading
from cv_web_app import app as flask_app

def run_flask():
    flask_app.run(port=5000, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()

st.components.v1.iframe(
    "https://localhost:5000",
    height=800
)