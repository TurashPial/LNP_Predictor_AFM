#  LNP Predictor Toolkit
This toolkit predicts lipid nanoparticle (LNP) characteristics based on user inputs using pretrained machine learning models.

## Folder Contents
- `read_trained_predict_modified.py`: Python script to collect input and make predictions, to local run
- `LNP_Predictor.ipynb`: Google Colab-compatible notebook
- `lnp_radius_vs_time.csv`: Time-vs-radius data used for interpolating R₀ from mixing time
- `*.pkl`: Trained Random Forest models and scalers

##How to Use 
### In Google Colab, if you don't want to run locally:
1. Upload this folder to Google Drive, e.g., `MyDrive/LNP_Predictor_Colab/`
2. Open `LNP_Predictor.ipynb` in Google Colab (double clicks autometicaaly opens in colab for this file type)
3. Follow prompts in the notebook or run the script manually:
```python
import read_trained_predict_modified as lnp
lnp.run_prediction()
```

## Input Parameter Ranges
- **Flow Rate**: ≥ 10 mL/min
- **PEG Molecular Weight**: 1000 to 5000 Da
- **PEG/Lipid Ratio**: 1% to 3% (0.01 to 0.03)
- **Salt Concentration**: 10 mM to 150 mM (0.01 to 0.15 Mol)

## Outputs
- **Predicted LNP Radius** (nm)
- **Empty LNP Ratio**
- **RNA Payload CV**

Ensure inputs are within these ranges for reliable predictions.
