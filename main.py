import pickle
import streamlit as st
import pandas as pd
import sklearn
import numpy

st.set_page_config(page_title='IPL | PRIDICTOR', page_icon='favicon.ico')

pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Delhi Capitals',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

st.title("IPL Win Predictor")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox("Select the batting team", teams)

with col2:
    bolling_team = st.selectbox("Select the bolling team", teams)

city = st.selectbox("Select the city", cities)

target = st.number_input("Target")

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input("Score")

with col4:
    over_completed = st.number_input("Over Completed")

with col5:
    wicket_out = st.number_input("Wicket Out")

if st.button("Predict Probablity"):
    runs_left = target - score
    balls_left = 120 - (over_completed*6)
    wicket_out = 10 - wicket_out
    crr = score/over_completed
    rrr = (runs_left*6)/balls_left

    input_data = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bolling_team],
        'city':[city],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wicket':[wicket_out],
        'total_runs_x':[target],
        'crr':[crr],
        'rrr':[rrr]
    })

    result = pipe.predict_proba(input_data)

    st.subheader(f'{bolling_team} : {round(result[0][0]*100, 3)}%')
    st.subheader(f'{batting_team} : {round(result[0][1]*100, 3)}%')



