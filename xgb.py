import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('C:/Users/kunji/OneDrive/Desktop/data/augmented_arecanut_dataset5k.csv')

# Preprocessing function to convert scientific notation
def convert_scientific_notation(x):
    if isinstance(x, str) and 'x 10' in x:
        parts = x.split('x 10^')
        return float(parts[0]) * (10 ** int(parts[1]))
    return float(x)

# Convert scientific notation to float
df['Beneficial_Microbes (CFU/g)'] = df['Beneficial_Microbes (CFU/g)'].apply(convert_scientific_notation)
df['Harmful_Microbes (CFU/g)'] = df['Harmful_Microbes (CFU/g)'].apply(convert_scientific_notation)
df['Soil_Organic_Carbon'] = df['Soil_Organic_Carbon'].str.replace('%', '').astype(float) / 100.0

# Define numeric columns
numeric_cols = [
    'Soil_pH', 'N (Nitrogen)', 'P (Phosphorus)', 'K (Potassium)', 
    'Organic_Matter (kg compost)', 'Temperature (°C)', 'Rainfall (mm)', 
    'Elevation (m)', 'Beneficial_Microbes (CFU/g)', 'Harmful_Microbes (CFU/g)', 
    'Microbial_Biomass_C (g/kg)', 'Soil_Organic_Carbon'
]

# Convert to numeric and handle errors
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Fill missing values with column means
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split data into features and target variable, excluding Sample_ID
X = df.drop(['Crop_Yield (kg/palm)', 'Sample_ID'], axis=1)
y = df['Crop_Yield (kg/palm)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the XGBoost model
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

# Fit the model
xgb_model.fit(X_train_scaled, y_train)

# Predicting the yields for test data
y_pred = xgb_model.predict(X_test_scaled)

# Calculate and print performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Feature Importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# Optionally: Plotting feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from XGBoost')
plt.gca().invert_yaxis()
plt.show()

joblib.dump(xgb_model, 'xgboost_model.pkl')

print("Model saved as 'xgboost_model.pkl'")


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, 'minmax_scaler.pkl')

print("Models saved successfully.")