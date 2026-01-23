# LSTM-Transformer Hybrid Model for Financial Forecasting

This project implements a hybrid deep learning model (LSTM + Transformer) to predict the daily directional return of financial assets (like the S&P 500).

## 🚀 Execution Order

Follow these steps to run the pipeline from start to finish:

### 1. Data Acquisition
Download the latest historical data from Yahoo Finance.
```powershell
.\venv\Scripts\python.exe src/data/loader.py
```
*Outputs: `data/raw/GSPC_raw.parquet`*

### 2. Training
Process data, generate features, and train the model.
```powershell
.\venv\Scripts\python.exe -m src.train
```
*Outputs: `best_model.pth` (Model weights), `data/scaler.save` (Feature Scaler)*

### 3. Evaluation (Optional)
Evaluate the model performance on the hold-out Test Set (2025).
```powershell
.\venv\Scripts\python.exe src/evaluate.py
```
*Outputs: Performance metrics (MSE, Directional Accuracy) in the console.*

### 4. Deployment (Web Interface)
Launch the Streamlit web app to make live predictions.
```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```
*Outputs: Starts a local web server (usually at http://localhost:8501).*

## 📂 Project Structure & File Roles

### 🟢 Entry Points (Run these!)
These files connect the components and execute the Phase logic.
- **`src/data/loader.py`**: Downloads the raw data from Yahoo Finance.
- **`src/train.py`**: The main brain. Orchestrates data loading, feature engineering, and model training.
- **`src/evaluate.py`**: Your quality control. Checks how well the model predicts 2025.
- **`app.py`**: The interface. Displays the predictions in a web browser.

### 🔵 Internal Modules (Do not run manually)
These are "Helper Files" that contain the logic classes. They are automatically imported by the Entry Points above.
- `src/data/preprocessor.py`: Logic for splitting (2019-2023 vs 2024) and scaling.
- `src/data/window_generator.py`: Logic for creating the 60-day sliding windows.
- `src/features/engineering.py`: Math for RSI, MACD, and Log-Returns.
- `src/inference.py`: The prediction engine used purely by the `app.py`.

