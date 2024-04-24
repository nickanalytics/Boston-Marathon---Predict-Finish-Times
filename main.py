import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import joblib
import datetime
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Boston Marathon", page_icon=":sports_medal:",layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)

# Load the models
model_lr_5k = joblib.load('models\model_lr_5k.pkl')
model_lr_20k = joblib.load('models\model_lr_20k.pkl')
model_lr_35k = joblib.load('models\model_lr_35k.pkl')

# Load the scalers
scaler_lr_5k = joblib.load('models\scaler_lr_5k.pkl')
scaler_lr_20k = joblib.load('models\scaler_lr_20k.pkl')
scaler_lr_35k = joblib.load('models\scaler_lr_35k.pkl')

col0_1, col0_2, col0_3 = st.columns([1, 6, 1])

with col0_2:
    st.markdown("""
                <h1 style='text-align: center;'>&#x1F3C5; 2023 Boston Marathon Statistics</h1>
                <h3 style="text-align: center; color: gray;">--- by Nick Analytics ---</h3>""", unsafe_allow_html=True)

# Draw a line
st.markdown("------------------------------------------------")

col1_1, col1_2 = st.columns((2),gap="medium")

with col1_1:
    # Apply custom font size using HTML and markdown
    st.markdown("""
    <h2 style='text-align: left; font-size: 20px;'>Boston Marathon Finish Time Predictor</h2>
    """, unsafe_allow_html=True)
    st.write("Please enter the following details:")

    # function that converts minutes to hh:mm:ss format
    def convert_to_hms(minutes):
        # Convert minutes to seconds
        seconds = minutes * 60
        # Convert seconds to a timedelta object
        timedelta_obj = datetime.timedelta(seconds=seconds)
        # Extract hours, minutes, and seconds
        hours, remainder = divmod(timedelta_obj.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        # Format the output as "3h : 54m : 59s"
        hms_format = f"{hours}h : {minutes}m : {seconds}s"
        return hms_format
    
    # Select passing point
    passing_point = st.selectbox("Choose the passing point", options=['5k', '20k', '35k'])
    
    if passing_point == '5k':
        # 5k input
        st.markdown("""<h2 style='text-align: left; font-size: 20px;'>Enter the 5k Passing Time (in minutes)</h2>""", unsafe_allow_html=True)
        minutes_5k = st.number_input("5k passing time in minutes (e.g. 23.00)")
        # gender_M_5k = st.selectbox("Gender (Male=1, Female=0)", options=[0, 1])
        gender_M_5k = 0 if st.radio("Gender for 5k", options=["Male", "Female"], index=0) == "Male" else 1
    
    elif passing_point == '20k':
        # 20k input
        st.markdown("""<h2 style='text-align: left; font-size: 20px;'>Enter the 15k & 20k Passing Time (in minutes)</h2>""", unsafe_allow_html=True)
        minutes_15k = st.number_input("15k passing time in minutes (e.g. 75.00)")
        minutes_20k = st.number_input("20k passing time in minutes (e.g. 100.00)")
        # gender_M_20k = st.selectbox("Gender (Male=1, Female=0)", options=[0, 1], key='gender_20k')
        gender_M_20k = 0 if st.radio("Gender for 20k", options=["Male", "Female"], index=0, key='gender_20k') == "Male" else 1
    
    elif passing_point == '35k':
        # 35k input
        st.markdown("""<h2 style='text-align: left; font-size: 20px;'>Enter the 30k & 35k Passing Time (in minutes)</h2>""", unsafe_allow_html=True)
        minutes_30k = st.number_input("30k passing time in minutes (e.g. 150.00)")
        minutes_35k = st.number_input("35k passing time in minutes (e.g. 180.00)")
        # gender_M_35k = st.selectbox("Gender (Male=1, Female=0)", options=[0, 1], key='gender_35k')
        gender_M_35k = 0 if st.radio("Gender for 35k", options=["Male", "Female"], index=0, key='gender_35k') == "Male" else 1


    # Predict button and results display logic
    if st.button("Predict"):
        predicted_time = None
        if passing_point == '5k':
            input_data_5k_base = np.array([[minutes_5k, gender_M_5k]])
            input_data_lr_5k = np.array([np.append(input_data_5k_base[0], [1 if gender_M_5k == 0 else 0, round(5 * (60/input_data_5k_base[0][0]),2)])])
            input_df_lr_5k = pd.DataFrame(input_data_lr_5k, columns=['5k_minutes','gender_M', 'gender_F','avg_pace_to_5k'])
            input_data_scaled_lr_5k = scaler_lr_5k.transform(input_df_lr_5k)
            predicted_time = model_lr_5k.predict(input_data_scaled_lr_5k)[0]
            
        elif passing_point == '20k':
            input_data_20k_base = np.array([[minutes_15k, minutes_20k, gender_M_20k]])
            perc_decay_15k_to_20k = round(((20 / minutes_20k) - (15 / minutes_15k)) / (15 / minutes_15k) * 100, 2)
            input_data_lr_20k = np.array([np.append(input_data_20k_base[0][:2], [perc_decay_15k_to_20k, gender_M_20k, 1 if gender_M_20k == 0 else 0])])
            input_df_lr_20k = pd.DataFrame(input_data_lr_20k, columns=['15k_minutes','20k_minutes','perc_decay_15k_to_20k','gender_M', 'gender_F'])
            input_data_scaled_lr_20k = scaler_lr_20k.transform(input_df_lr_20k)
            predicted_time = model_lr_20k.predict(input_data_scaled_lr_20k)[0]
            
        elif passing_point == '35k':
            input_data_35k_base = np.array([[minutes_30k, minutes_35k, gender_M_35k]])
            perc_decay_30k_to_35k = round(((35 / minutes_35k) - (30 / minutes_30k)) / (30 / minutes_30k) * 100, 2)
            input_data_lr_35k = np.array([np.append(input_data_35k_base[0][:2], [perc_decay_30k_to_35k, gender_M_35k, 1 if gender_M_35k == 0 else 0])])
            input_df_lr_35k = pd.DataFrame(input_data_lr_35k, columns=['30k_minutes','35k_minutes','perc_decay_30k_to_35k','gender_M','gender_F'])
            input_data_scaled_lr_35k = scaler_lr_35k.transform(input_df_lr_35k)
            predicted_time = model_lr_35k.predict(input_data_scaled_lr_35k)[0]

        if predicted_time is not None:
            predicted_time_hms = convert_to_hms(predicted_time)
            st.metric(label="Predicted Finish Time", value=predicted_time_hms)

with col1_2:
    mae = pd.read_csv('data/RMSE and MAE of the marathon models.csv')
    st.markdown("""<h2 style='text-align: left; font-size: 20px;'>How much on average does the prediction model deviate (in minutes)</h2>""", unsafe_allow_html=True)
    fig = px.line(mae, x="distance", y="MAE", labels={"MAE": "Avg deviation in minutes", "distance":"passing point"}, height=400, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    # Add text under the plot
    st.text('Predictions created by the model become increasingly precise when we get closer to the finish.')
    st.text('After 30k the model deviates less than 2 minutes from the actual finish time')

st.markdown("------------------------------------------------")
    
col2_1, col2_2, col2_3 = st.columns([2, 4, 4])

@st.cache_data
def load_gender():
    raw_df = pd.read_csv('data/raw_data_2023.csv')
    raw_gender = raw_df['gender']
    return raw_gender

def plot_top_left():
    df_gender = load_gender()
        
    # Calculate the count of males and females
    df_gender = df_gender.value_counts().reset_index()
    df_gender.columns = ['gender', 'count']
    
    # Replace 'M' and 'F' with 'Male' and 'Female'
    df_gender['gender'] = df_gender['gender'].replace({'M': 'Male', 'F': 'Female'})
    
    # Create the bar plot with go.Bar
    fig = go.Figure(data=[
        go.Bar(name='Males vs. Females', x=df_gender['gender'], y=df_gender['count'], text=df_gender['count'], width=0.5)
    ])
    
    fig.update_layout(
        title="Male & Female Runners",
        title_x=0.25,  # this line centers the title
        yaxis_title="Count",
        height=400,
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1  # gap between bars of the same location coordinates
    )
    
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


with col2_1:
    plot_top_left()
    
def plot_lines():
    pace_df = pd.read_csv('data/df_gender_pace22.csv')
    fig = go.Figure()

    for gender in pace_df['gender'].unique():
        df_gender = pace_df[pace_df['gender'] == gender]
        if gender == 'female':  # Make sure this matches the gender value in your data
            fig.add_trace(go.Scatter(x=df_gender['age'], y=df_gender['avg_pace'], mode='lines', name=gender, line=dict(color='rgb(255,0,0)')))
        else:
            fig.add_trace(go.Scatter(x=df_gender['age'], y=df_gender['avg_pace'], mode='lines', name=gender))

    fig.update_layout(title='Average Pace per Age/Gender',title_x=0.35, xaxis_title='Age', yaxis_title='Average Pace',height=400)
    st.plotly_chart(fig)

with col2_2:
    plot_lines()


def plot_top_right():
    # Read mean paces from the CSV file
    mean_paces = pd.read_csv('data/mean_pace.csv')
    
    # Prepare data by resetting the index to make the checkpoints and mean paces columns
    mean_paces = mean_paces.melt(var_name='Checkpoint', value_name='Mean Pace (km/h)')
    
    # Create a line plot using Plotly
    fig = px.line(
        mean_paces,
        x='Checkpoint',
        y='Mean Pace (km/h)',
        title='Mean Pace (km/h) at Each Checkpoint',
        markers=True
    )
    
    # Update layout for better visualization
    fig.update_layout(
        xaxis_title="Checkpoint",
        yaxis_title="Mean Pace (km/h)",
        xaxis={'type': 'category'},  # This makes sure the x-axis is treated as categorical
        title_x=0.35,  # Center the title
        height=400
    )
    
    # Display the plot in the specified Streamlit column
    st.plotly_chart(fig, use_container_width=True)

# Integrate the plot into the Streamlit layout
with col2_3:
    plot_top_right()

st.markdown(
    '<hr style="border:1px solid #ccc; margin-top: 0.5rem; margin-bottom: 0.5rem;"/>',
    unsafe_allow_html=True
)

# Define the columns for the third row
col3_1, col3_2 = st.columns((2), gap="medium")
@st.cache_data
def load_data():
    df = pd.read_csv('data/all2023new_no_outlier.csv')
    df_finish = df['finish_minutes']
    return df_finish

# Column 3-1: Finish Times Display
with col3_1:
    df_finish = load_data()
    st.markdown("""<h2 style='text-align: left; font-size: 20px;'>Finish Times (minutes)</h2>""", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_finish, bins=50, ax=ax)
    ax.set_xlabel('Finish Times in Minutes')
    ax.set_ylabel('Number of Runners')
    mean = df_finish.mean()
    plt.axvline(mean, color='r', linestyle='dotted', linewidth=2)
    ax.text(mean + ax.get_xlim()[1]*0.01, ax.get_ylim()[1]*0.9, f'Mean: {mean:.2f}', color='r', ha='left')
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    st.pyplot(fig, use_container_width=True)
    st.text('Distribution of finish times in 2023. Mean finish time for all runners is 03:39:12 hrs.')

# Column 3-2: Data Sorted by Standard Deviation
with col3_2:
    st.markdown("""<h2 style='text-align: left; font-size: 20px;'>Data Sorted by Standard Deviation</h2>""", unsafe_allow_html=True)
    try:
        data = pd.read_csv('data/avg_std_small.csv')
        st.dataframe(data.head(15), hide_index=True)  # Displaying the DataFrame in the correct column
        st.text('The runner with the flattest run has a pace variation of less that 0,1 km/hr over the entire distance.')
    except Exception as e:
        st.error(f"Failed to load and sort data: {str(e)}")

