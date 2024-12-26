import streamlit as st
import pandas as pd
import joblib

#load the trained model
model = joblib.load('liver_cirrhosis_model.pkl')

#define the column names for input (using only important columns)
columns = ['N_Days', 'Status', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders',
           'Edema', 'Bilirubin', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
           'Prothrombin']

def get_user_input():
# 3 rows with 3 columns eac are created as streamlit columns
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row1_col1:
        N_Days = st.number_input('N_Days', min_value=0)
    with row1_col2:
        Status = st.selectbox('Status', options=[0, 1, 2], index=0)
    with row1_col3:
        Sex = st.selectbox('Sex', options=[0, 1], index=0)
    with row2_col1:
        Ascites = st.selectbox('Ascites', options=[0, 1], index=0)
    with row2_col2:
        Hepatomegaly = st.selectbox('Hepatomegaly', options=[0, 1], index=0)
    with row2_col3:
        Spiders = st.selectbox('Spiders', options=[0, 1], index=0)
    with row3_col1:
        Edema = st.selectbox('Edema', options=[0, 1, 2], index=0)
    with row3_col2:
        Bilirubin = st.number_input('Bilirubin', min_value=0.0)
    with row3_col3:
        Albumin = st.number_input('Albumin', min_value=0.0)

    #remaining inputs
    Copper = st.number_input('Copper', min_value=0.0)
    Alk_Phos = st.number_input('Alk_Phos', min_value=0.0)
    SGOT = st.number_input('SGOT', min_value=0.0)
    Prothrombin = st.number_input('Prothrombin', min_value=0.0)

    #input dataframe
    user_data = [[N_Days, Status, Sex, Ascites, Hepatomegaly, Spiders,
                  Edema, Bilirubin, Albumin, Copper, Alk_Phos, SGOT, Prothrombin]]
    user_data_df = pd.DataFrame(user_data, columns=columns)

    return user_data_df

def main():
    st.title('Liver Cirrhosis Stage Prediction')
    #take input
    user_input = get_user_input()
    #predict
    prediction = model.predict(user_input)
    #output!!
    st.write(f"The predicted stage of liver cirrhosis is: {prediction[0]}")

if __name__ == "__main__":
    main()
