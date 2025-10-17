import streamlit as st
import sys
import os

# Simple test app to isolate JavaScript issues
st.set_page_config(
    page_title="Test App",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 EmpowerHer Chatbot - Test Mode")

st.success("✅ Streamlit is working correctly!")

st.markdown("""
This is a minimal test version to check if the basic Streamlit functionality works.

If you can see this page, the JavaScript syntax error is resolved.
""")

# Test basic functionality
if st.button("Test Button"):
    st.balloons()
    st.success("Button works!")

# Test text input
user_input = st.text_input("Test Input", placeholder="Type something...")
if user_input:
    st.write(f"You typed: {user_input}")

st.info("If this page loads without JavaScript errors, we can proceed with the full app.")
