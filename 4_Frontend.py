import streamlit as st
from streamlit_option_menu import option_menu
import requests
import pandas as pd

st.set_page_config(page_title="Churn Analysis Dashboard", layout="wide")

theme = st.sidebar.selectbox('Theme', ['light', 'dark'])
if theme == 'dark':
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

selected = option_menu(
    menu_title=None,
    options=["Prediction", "Text2SQL", "RAG"],
    icons=["graph-up", "chat-dots", "search"],
    orientation="horizontal",
)

if selected == "Prediction":
    st.title("Customer Churn Prediction")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            senior_citizen = st.selectbox("Senior Citizen", ["False", "True"], format_func=lambda x: x)
            senior_citizen = 1 if senior_citizen == "True" else 0
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure", min_value=0)
            online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
            
        with col2:
            online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
            tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            
        with col3:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", 
                ["Bank transfer (automatic)", "Credit card (automatic)", 
                 "Electronic check", "Mailed check"])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
            total_charges = st.number_input("Total Charges", min_value=0.0)
            
        submitted = st.form_submit_button("Predict Churn")
        if submitted:
            # Convert categorical variables to numeric
            partner_encoded = 1 if partner == "Yes" else 0
            dependents_encoded = 1 if dependents == "Yes" else 0
            online_security_encoded = 0 if online_security == "No" else 1 if online_security == "No internet service" else 2
            online_backup_encoded = 0 if online_backup == "No" else 1 if online_backup == "No internet service" else 2
            device_protection_encoded = 0 if device_protection == "No" else 1 if device_protection == "No internet service" else 2
            tech_support_encoded = 0 if tech_support == "No" else 1 if tech_support == "No internet service" else 2
            contract_encoded = 0 if contract == "Month-to-month" else 1 if contract == "One year" else 2
            paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0
            payment_method_encoded = {
                "Bank transfer (automatic)": 0,
                "Credit card (automatic)": 1,
                "Electronic check": 2,
                "Mailed check": 3
            }[payment_method]
            
            # Create feature array for prediction
            features = [
                senior_citizen, partner_encoded, dependents_encoded, tenure,
                online_security_encoded, online_backup_encoded, device_protection_encoded,
                tech_support_encoded, contract_encoded, paperless_billing_encoded,
                payment_method_encoded, monthly_charges, total_charges
            ]
            
            response = requests.post(
                "http://localhost:8000/predict",
                json={"features": features}
            )
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                probability = prediction * 100
                st.success(f"Churn Probability: {probability:.2f}%")
            else:
                st.error("Error getting prediction")
                    
elif selected == "Text2SQL":
    st.title("SQL Query Generator")
    query = st.text_input("Enter your question about the data:")
    
    if st.button("Generate SQL"):
        if query:
            try:
                response = requests.post(
                    "http://localhost:8000/generate_sql",
                    json={"query": query}
                )
                response.raise_for_status()
                result = response.json()
                
                st.success("SQL Generated Successfully!")
                st.code(result["sql"], language="sql")
                
                if result["results"]:
                    st.subheader("Query Results")
                    # Convert results to pandas DataFrame for better display
                    df = pd.DataFrame(result["results"])
                    st.dataframe(df)
                    
                    # Add download button for CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Query executed successfully but returned no results")
                    
                with st.expander("View Raw Response"):
                    st.json(result)
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {str(e)}")
                if hasattr(e.response, 'json'):
                    st.json(e.response.json())
        else:
            st.warning("Please enter a question first")

elif selected == "RAG":
    st.title("Knowledge Base Query")
    query = st.text_input("Enter your question:")
    
    if st.button("Search"):
        if query:
            try:
                response = requests.post(
                    "http://localhost:8000/rag_query",
                    json={"query": query}
                )
                response.raise_for_status()
                result = response.json()
                
                st.success("Query Processed Successfully!")
                
                # Display the answer
                st.subheader("Answer")
                st.write(result["answer"])
                
                # Display relevant sources/documents
                if result.get("sources"):
                    with st.expander("View Source Documents"):
                        for idx, source in enumerate(result["sources"], 1):
                            st.markdown(f"**Source {idx}:**")
                            st.write(source)
                            st.markdown("---")
                
                # Display confidence score if available
                if "confidence" in result:
                    st.info(f"Confidence Score: {result['confidence']:.2f}")
                
                with st.expander("View Raw Response"):
                    st.json(result)
            except requests.exceptions.RequestException as e:
                st.error(f"Error: {str(e)}")
                if hasattr(e.response, 'json'):
                    st.json(e.response.json())
        else:
            st.warning("Please enter a question first")