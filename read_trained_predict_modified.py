import pandas as pd
import numpy as np
import joblib
import os
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#  Constants 
b = 1.3e3
alpha = 1.520
MODEL_DIR = "."
FEATURE_NAMES = ["R0", "RF", "fiPEG0", "csalt"]
TARGETS = ["size", "empty", "RNA_pooled_CV"]

# Functions 
def convert_mw_to_rf(mw):
    a = 0.37  # nm
    monomer_weight = 44  # g/mol
    n = mw / monomer_weight
    return a * n**(3/5)

def get_tauM_from_Q(Q):
    return (b / 1.75) * Q**(-alpha) / 1000  # in seconds

def get_R0_from_tauM(tau_M):
    try:
        df = pd.read_csv("lnp_radius_vs_time.csv")
    except FileNotFoundError:
        print("'lnp_radius_vs_time.csv' not found.")
        exit()

    time_s = df["time_s"].values
    radius_nm = df["radius_nm"].values  

    if tau_M < time_s.min() or tau_M > time_s.max():
        print(f" τ_M = {tau_M:.6e} s is outside simulation range.")
        print(f"   Simulation covers: [{time_s.min():.2e}, {time_s.max():.2e}] s")
        exit()

    interp_R = interp1d(time_s, radius_nm, kind="linear")
    return float(interp_R(tau_M))

def get_user_input():
    print("Enter experimental conditions:")
    try:
        Q = float(input("Flow rate Q (mL/min): "))
        MW = float(input("PEG molecular weight (Da): "))
        fiPEG0 = float(input("PEG/Lipid Ratio (e.g. 0.015): "))
        csalt = float(input("Salt concentration (Mol): "))

        if Q <= 0 or MW <= 0 or fiPEG0 <= 0 or csalt <= 0:
            raise ValueError
    except ValueError:
        print(" Invalid input. Please enter positive numerical values.")
        exit()

    tau_M = get_tauM_from_Q(Q)
    print(f" τ_M (charachteristic mixing time) = {tau_M:.6f} s")

    R0 = get_R0_from_tauM(tau_M)
    print(f" Interpolated LNP Radius at τ_M: {R0:.4f} nm")

    RF = convert_mw_to_rf(MW)
    print(f" Flory radius RF = {RF:.4f} nm")

    return [R0, RF, fiPEG0, csalt]

def predict_all(features):
    print("\n Predicting LNP properties...")

    results = {}
    for target in TARGETS:
        model_path = os.path.join(MODEL_DIR, f"rf_model_{target}.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{target}.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f" Missing model or scaler for '{target}'.")
            results[target] = "Missing"
            continue

        rf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        X_new = pd.DataFrame([features], columns=FEATURE_NAMES)
        X_scaled = scaler.transform(X_new)

        prediction = rf.predict(X_scaled)[0]
        results[target] = round(prediction, 4)

    input_summary = {
        "Flow Rate (mL/min)": features[0],
        "PEG MW (Da)": round((features[1] / 0.37)**(5/3) * 44, 1),
        "PEG/Lipid Ratio": features[2],
        "Salt Concentration (Mol)": features[3]
    }

    summary_df = pd.DataFrame({
        "Parameter": list(input_summary.keys()) + list(results.keys()),
        "Value": list(input_summary.values()) + list(results.values())
    })

    print("\n Summary Table:")
    print(summary_df)  # Use display() only in notebooks; use print(summary_df) in CLI


# === Main Execution ===
def run_prediction():
    features = get_user_input()
    predict_all(features)


if __name__ == "__main__":
    run_prediction()
