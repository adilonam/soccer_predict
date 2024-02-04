

import streamlit as st
import pandas as pd
import joblib
import json
from tensorflow.keras.models import load_model 


def main():
    loaded_model = load_model('./model_data/modeltrained_model.keras')
    loaded_scaler = joblib.load('./model_data/scaler.joblib')

    with open("./model_data/team_id.json", 'r') as file:
            team_id = json.load(file)
    with open("./model_data/win_rate.json", 'r') as file:
            win_rate = json.load(file)


    st.title("Two Team Prediction")

    # Input for the first value
    input_value1 = st.text_input("Enter the first team:")

    # Input for the second value
    input_value2 = st.text_input("Enter the second team:")

    # Display the entered values
    st.write(f"You entered the first value: {input_value1}")
    st.write(f"You entered the second value: {input_value2}")

    st.write(f"team ids : {team_id}")
    


    # Button to trigger an action
    if st.button("Predict"):
        #get df test
        test = {}
        test['team_home'] = input_value1
        test['team_away'] = input_value2
        test['win_rate_team1'] = win_rate[test['team_home']]
        test['win_rate_team2'] = win_rate[test['team_away']]


        df_test = pd.DataFrame(test, index=[0])

        x_test = loaded_scaler.transform(df_test)
        predictions = loaded_model.predict(x_test)
        
        result = "WIN" if predictions >= 0.5 else "lose"
        st.write(f"Result : {result}")
        
        prc =round((abs(predictions[0,0]-0.5)+0.5)*100,2)
        st.write(f"Probability : {prc}%")
    


if __name__ == "__main__":
    main()