# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import gdown

# **Step 1: Load Dataset**
file_id = "1ZbR2jioaBxLJEBHQ3OWdbOcRBrdKjRx2"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "dataset.csv", quiet=False)

df = pd.read_csv("dataset.csv")
#file_path = "indian_coal_mining_updated.csv"
#df = pd.read_csv(file_path)

# **Step 2: Remove Outliers (Improves Accuracy)**
for col in ["Carbon_Credits_Earned"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# **Step 3: Define Feature Selection Function**
def prepare_data(model_name):
    feature_sets = {
        "carbon_credit": {
            "X": ["CO2_Emissions_MtCO2e", "Reduction", "Mitigation_Strategies", "Coal_Consumption_Mt",
                  "FuelType", "Excavation", "Transportation", "Fuel", "Equipment", "Workers"],
            "y": "Carbon_Credits_Earned"
        }
    }

    if model_name not in feature_sets:
        raise ValueError("Invalid model name!")

    features = feature_sets[model_name]["X"]
    target = feature_sets[model_name]["y"]

    # Select relevant features
    X = df[features]
    y = df[target]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing pipeline
    num_pipeline = StandardScaler()
    cat_pipeline = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numerical_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    # Split data into train-test sets (85-15 split for better generalization)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Fit and transform preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Save preprocessor for future predictions
    joblib.dump(preprocessor, "carbon_credit_preprocessor.pkl")

    return X_train, X_test, y_train, y_test

# **Step 4: Train High Accuracy Carbon Credit Estimation Model**
def train_carbon_credit_model():
    print("🔹 Training Carbon Credit Estimation Model...")

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data("carbon_credit")

    # **Optimized XGBoost Model (Better Hyperparameters)**
    model = XGBRegressor(n_estimators=1000, learning_rate=0.02, max_depth=10, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, reg_lambda=1.2, reg_alpha=0.5)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "carbon_credit_xgb.pkl")

    print("✅ Model trained and saved successfully!")

    # Evaluate Model
    y_pred = model.predict(X_test)
    print(f"\n🔹 **Model Performance:**")
    print(f"📉 Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"📉 Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# **Step 5: Take User Input for Prediction**
def get_user_input():
    print("\n🔹 Enter mining operation details:")
    co2_emission = float(input("Predicted CO₂ Emissions (metric tons): "))
    reduction = float(input("Emissicdon Reduction (%): "))
    mitigation_strategy = input("Mitigation Strategy (Afforestation/Renewable Energy/etc.): ")
    coal_consumption = float(input("Coal Consumption (million tons): "))
    fuel_type = input("Fuel Type (Coal/Diesel/Gas): ")
    excavation = float(input("Excavation Volume (thousand m³): "))
    transportation = float(input("Transportation Distance (km): "))
    fuel = float(input("Fuel Consumption (thousand liters): "))
    equipment = float(input("Equipment Usage (hours): "))
    workers = int(input("Number of Workers: "))

    return pd.DataFrame([[co2_emission, reduction, mitigation_strategy, coal_consumption, fuel_type,
                          excavation, transportation, fuel, equipment, workers]],
                        columns=["CO2_Emissions_MtCO2e", "Reduction", "Mitigation_Strategies", "Coal_Consumption_Mt",
                                 "FuelType", "Excavation", "Transportation", "Fuel", "Equipment", "Workers"])

# **Step 6: Predict Carbon Credits for New Data**
def predict_carbon_credits():
    user_input = get_user_input()

    # Load preprocessor and model
    preprocessor = joblib.load("carbon_credit_preprocessor.pkl")
    model = joblib.load("carbon_credit_xgb.pkl")

    # Preprocess user input
    user_input_transformed = preprocessor.transform(user_input)

    # Predict Carbon Credits
    predicted_credit = model.predict(user_input_transformed)[0]

    print(f"\n🔹 **Estimated Carbon Credits Earned:** {predicted_credit:.2f}")

# **Step 7: Visualize Carbon Credit Trends**
def visualize_carbon_credit_trends():
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Carbon_Credits_Earned"], bins=20, kde=True, color="green")
    plt.title("Distribution of Carbon Credits Earned")
    plt.xlabel("Carbon Credits (Tons of CO₂ Equivalent)")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Mitigation_Strategies", y="Carbon_Credits_Earned", data=df, palette="Set2")
    plt.title("Carbon Credits by Mitigation Strategy")
    plt.xticks(rotation=45)
    plt.show()

# **Step 8: Run Program**
if __name__ == "__main__":
    train_carbon_credit_model()  # Train the model (Only run this once)
    predict_carbon_credits()  # Take user input and predict Carbon Credits
    visualize_carbon_credit_trends()  # Show Carbon Credit trends
