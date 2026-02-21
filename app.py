import pickle
import streamlit as st
import pandas as pd

model = pickle.load(open('model.pkl','rb'))
st.title('Loan Prediction app')
st.write('Enter customer detail : ')

col1,col2 = st.columns(2)

with col1:
    with st.form("loan_form"):

        dependents = st.slider("Number of Dependents", 0, 5)

        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed?", ["Yes", "No"])

        income = st.number_input("Annual Income (₹)", min_value=1)

        loan_amount = st.number_input("Loan Amount Required (₹)", min_value=0)

        loan_term = st.slider("Loan Term (Years)", 1, 30)

        cibil_score = st.slider("CIBIL Score", 300, 900)

        total_assets = st.number_input("Total Assets Value (₹)", min_value=1)

        submit = st.form_submit_button("Predict Loan Status")

    if submit:


        residential = total_assets * 0.25
        commercial = total_assets * 0.25
        luxury = total_assets * 0.25
        bank = total_assets * 0.25

        loan_to_income = loan_amount / income
        loan_to_asset = loan_amount / total_assets
        asset_to_income = total_assets / income

        #input_list = [['dependents','education','self_employed','income','loan_amount','loan_term',
        #    'cibil_score','residential','commercial','luxury','bank','total_assets','loan_to_income','loan_to_asset','asset_to_income']]

        input_data = pd.DataFrame([{
        "no_of_dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential,
        "commercial_assets_value": commercial,
        "luxury_assets_value": luxury,
        "bank_asset_value": bank,
        "total_assets_value": total_assets,
        "loan_to_income": loan_to_income,
        "loan_to_asset": loan_to_asset,
        "asset_to_income": asset_to_income
    }])

        
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ Loan Approved!")
        else:
            st.error("❌ Loan Rejected!")

with col2:
    with st.expander('Application summary'):
        R = 8 / (12 * 100)   # monthly interest
        N = loan_term * 12 
        emi = (loan_amount * R * (1 + R)**N) / ((1 + R)**N - 1)

        loan_to_income = loan_amount / income
        loan_to_asset  = loan_amount / total_assets
        asset_to_income = total_assets / income

        summary = {
                "Annual Income": f"₹{income:,.0f}",
                "Loan Amount":   f"₹{loan_amount:,.0f}",
                "Loan Term":     f"{loan_term} years",
                "Total Assets":  f"₹{total_assets:,.0f}",
                "Est. EMI":      f"₹{emi:,.0f}/month",
                "Loan-to-Income": f"{loan_to_income:.2f}x",
                "Loan-to-Asset":  f"{loan_to_asset*100:.1f}%",
                "Asset/Income":   f"{asset_to_income:.2f}x",
            }
        
        df_summary = pd.DataFrame(summary.items(), columns=["Field", "Value"])
        st.dataframe(df_summary, use_container_width=True, hide_index=True)