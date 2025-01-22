import streamlit as st
import numpy as np
from utils.predict import predict


# Streamlit app
def main():
    st.title("University Admission Prediction App")

    # Input fields
    GRE_score = st.number_input("Enter GRE Score", min_value=260, max_value=340, step=1)
    TOEFL_score = st.number_input("Enter TOEFL Score", min_value=0, max_value=120, step=1)
    CGPA = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.01)
    LOR = st.slider("Enter Letter of Recommendation (LOR) Rating", min_value=0.0, max_value=5.0, step=0.5)
    Research = st.selectbox("Do you have Research Experience?", options=["No", "Yes"])
    Research = 1 if Research == "Yes" else 0  # Convert to binary 1/0
    
    # Predict button
    if st.button("Predict"):
        input=np.array([[GRE_score, TOEFL_score, LOR, CGPA, Research]])
        result=predict(input)
        st.write(f"You have {round(result[0],2)*100}% chance of getting admit")

if __name__ == "__main__":
    main()
