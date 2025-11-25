📘 AI-Based Anomaly Detection on NASA Turbofan Engine Sensor Data (CMAPSS)

📌 Overview

This project identifies abnormal behavior in real turbofan aircraft engine telemetry using a hybrid ML + DL pipeline:

Isolation Forest — unsupervised anomaly detection

Autoencoder Neural Network — reconstruction-based anomaly detection

Dataset: NASA CMAPSS FD001, containing multivariate engine sensor recordings across operational cycles.

Used in aviation, defence telemetry, predictive maintenance, reliability engineering.

📂 Project Structure
anomaly_detection/
│
├── dataset/                 # Place NASA FD001 dataset here after download
│
├── src/
│   ├── preprocessing.py     # Data loading & normalization
│   ├── anomaly_detection.py # ML/DL anomaly detection pipeline
│   └── visualize.py         # Plots & analysis
│
├── results/
│   ├── anomaly_plot.png     # Final visualization
│   └── anomaly_log.txt      # Detected anomaly records
│
├── requirements.txt
└── README.md

🎯 Dataset Download (Required)

Dataset not included due to licensing and size limits.

Download from:
https://data.nasa.gov/dataset/CMAPSS/

Place FD001 file inside:

dataset/train_FD001.txt

🛠 Technologies Used

Python

NumPy, Pandas

Scikit-learn (Isolation Forest)

TensorFlow / Keras (Autoencoder)

Matplotlib

🚀 How It Works
✅ 1. Load & preprocess data

normalize, clean, smooth sensor values

✅ 2. Train Isolation Forest

detects statistical outliers/spikes

✅ 3. Train Autoencoder

learns normal behavior → high error = anomaly

✅ 4. Hybrid decision rule
If Isolation Forest OR Autoencoder flags anomaly → anomaly

✅ 5. Output generated

anomaly plot

anomaly log file

▶️ Run the Project
pip install -r requirements.txt
python src/anomaly_detection.py

📈 Sample Output

🎯 Why This Project Matters (DRDO Relevance)

Uses real aviation-grade sensor telemetry

Demonstrates predictive maintenance capability

Implements hybrid ML + DL fault detection

Applicable to aircraft engines, UAV propulsion, missile systems

Reflects analytical workflows used in defence research labs

📚 Future Enhancements

LSTM/GRU sequence-based anomaly detection

Sensor correlation & root-cause analysis

Real-time streaming anomaly monitoring

Explainability using SHAP / feature attribution

📜 License

MIT License — free to use and modify.

👤 Author

Sravani Teeda
CSE (AI & ML), 2026
Open to Research & Defence Internships