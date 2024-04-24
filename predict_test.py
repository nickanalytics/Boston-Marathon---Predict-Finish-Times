import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load the models
model_lr_5k = joblib.load('models\model_lr_5k.pkl')
model_lr_20k = joblib.load('models\model_lr_20k.pkl')
model_lr_35k = joblib.load('models\model_lr_35k.pkl')

# Load the scalers
scaler_lr_5k = joblib.load('models\scaler_lr_5k.pkl')
scaler_lr_20k = joblib.load('models\scaler_lr_20k.pkl')
scaler_lr_35k = joblib.load('models\scaler_lr_35k.pkl')

# ---------------------Data input-----------------------------
# Input data style 5k: ['5k_minutes','gender_M']
input_data_5k_base = np.array([[26.87, 0]])
# Input data style 10k: ['15k_minutes','20k_minutes','gender_M']
input_data_20k_base = np.array([[80.50,108.10, 0]])
# Input data style 15k: ['30k_minutes','35k_minutes','gender_M']
input_data_35k_base = np.array([[164.78,193.65, 0]])

# ---------------------Feature Generation with new input arrays-----------------------------
# 5k two new features and new array creation
avg_pace_to_5k = round(5 * (60/input_data_5k_base[0][0]),2)
gender_f = 1 if input_data_5k_base[0][1] == 0 else 0

input_data_lr_5k = np.array([np.append(input_data_5k_base[0], [gender_f, avg_pace_to_5k])])

# 20k two new features and new array creation
perc_decay_15k_to_20k = round(((20 / input_data_20k_base[0][1]) - (15 / input_data_20k_base[0][0])) / (15 / input_data_20k_base[0][0]) * 100, 2)
gender_f = 1 if input_data_20k_base[0][2] == 0 else 0

input_data_lr_20k = np.array([np.append(input_data_20k_base[0][:2], [perc_decay_15k_to_20k, input_data_20k_base[0][2], gender_f])])

# 35k two new features and new array creation
perc_decay_30k_to_35k = round(((35 / input_data_35k_base[0][1]) - (30 / input_data_35k_base[0][0])) / (30 / input_data_35k_base[0][0]) * 100, 2)
gender_f = 1 if input_data_35k_base[0][2] == 0 else 0

input_data_lr_35k = np.array([np.append(input_data_35k_base[0][:2], [perc_decay_30k_to_35k, input_data_35k_base[0][2], gender_f])])

# ---------------Transform & Scale Data------------------------------------------
# 5k: Create a DataFrame and scale the data
input_df_lr_5k = pd.DataFrame(input_data_lr_5k, columns=['5k_minutes','gender_M', 'gender_F','avg_pace_to_5k'])

# Scale the input data using the corresponding scaler
input_data_scaled_lr_5k = scaler_lr_5k.transform(input_df_lr_5k)

# 20k: Create a DataFrame and scale the data
input_df_lr_20k = pd.DataFrame(input_data_lr_20k, columns=['15k_minutes','20k_minutes','perc_decay_15k_to_20k','gender_M', 'gender_F'])

# Scale the input data using the corresponding scaler
input_data_scaled_lr_20k = scaler_lr_20k.transform(input_df_lr_20k)

# 35k: Create a DataFrame and scale the data
input_df_lr_35k = pd.DataFrame(input_data_lr_35k, columns=['30k_minutes','35k_minutes','perc_decay_30k_to_35k','gender_M','gender_F'])

# Scale the input data using the corresponding scaler
input_data_scaled_lr_35k = scaler_lr_35k.transform(input_df_lr_35k)

# ---------------Make Predictions------------------------------------------
# 5k: Make the prediction
prediction_lr_5k = model_lr_5k.predict(input_data_scaled_lr_5k)

# Display the prediction
print("Prediction:", prediction_lr_5k)

# 20k: Make the prediction
prediction_lr_20k = model_lr_20k.predict(input_data_scaled_lr_20k)

# Display the prediction
print("Prediction:", prediction_lr_20k)

# 35k: Make the prediction
prediction_lr_35k = model_lr_35k.predict(input_data_scaled_lr_35k)

# Display the prediction
print("Prediction:", prediction_lr_35k)