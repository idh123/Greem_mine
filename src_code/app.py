from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load your trained model and preprocessor
model = joblib.load("co2_emission_model.pkl")
preprocessor = joblib.load("co2_emission_preprocessor.pkl")

# Load dataset once for graph generation
df = pd.read_csv("dataset.csv")

@app.route("/")
def home():
    return render_template("emissions.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Create DataFrame from input
        input_df = pd.DataFrame([[
            data["excavation"],
            data["distance"],          # ✅ match key from emissions.html
            data["fuel"],
            data["equipment"],
            data["workers"],
            data["fuel_type"],         # ✅ match key from emissions.html
            data["reduction"]
        ]], columns=["Excavation", "Transportation", "Fuel", "Equipment", "Workers", "FuelType", "Reduction"])

        # Transform input and predict
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)[0]

        # Save emission histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df["CO2_Emissions_MtCO2e"], bins=20, kde=True, color="blue")
        plt.title("Distribution of CO₂ Emissions")
        plt.xlabel("CO₂ Emissions (MtCO2e)")
        plt.ylabel("Frequency")
        hist_path = "static/emission_hist.png"
        plt.savefig(hist_path)
        plt.close()

        # Save fuel-type boxplot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="FuelType", y="CO2_Emissions_MtCO2e", data=df, palette="Set2")
        plt.title("CO₂ Emissions by Fuel Type")
        box_path = "static/emission_box.png"
        plt.savefig(box_path)
        plt.close()

        # Return prediction and image paths
        return jsonify({
            "emission": round(prediction, 2),
            "graph1": f"/{hist_path}",
            "graph2": f"/{box_path}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
