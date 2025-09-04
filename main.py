import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# 1. Problem Definition
# The goal of this project is to predict the Air Quality Index (AQI) based on pollutants like PM2.5, PM10, NO2, SO2, CO, and Temperature.

# 2. Data Collection
# Load the dataset
df = pd.read_csv('updated_pollution_dataset.csv')
print("Columns in the dataset:", df.columns)

# 3. Data Cleaning & Preprocessing
# Visualizing missing values before cleaning
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Before Cleaning')
plt.show()

# Drop rows with missing values and irrelevant columns (e.g., O3)
df_before_cleaning = df.copy()  # Save a copy for comparison
df = df.dropna()
df = df.drop(columns=['O3'], errors='ignore')

# Visualizing missing values after cleaning
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values After Cleaning')
plt.show()

print("Columns after cleaning:", df.columns)

# 4. Feature Engineering
# Define target column and features
target_column = 'Air Quality'
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Temperature']
X = df[features]
y = df[target_column]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 5. Model Selection
# RandomForestRegressor is chosen for this regression task

# 6. Model Training
# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Model Evaluation
# Predictions and evaluation
y_pred = model.predict(X_test)

# Model evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print evaluation results
print(f"R² score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")

# 8. Model Tuning (Optional)
# Hyperparameter tuning can be performed here if needed.

# 9. Model Deployment
# Save the trained model, scaler, and label encoder
joblib.dump(model, 'air_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Confirmation message
print("Model, Scaler, and Label Encoder saved successfully.")

# 10. Monitor and Maintain
# The model will be monitored and retrained as needed with new data.
