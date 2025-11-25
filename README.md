📘 AI-Based Anomaly Detection on NASA Turbofan Engine Sensor Data (CMAPSS)
📌 Overview

This project identifies abnormal behavior in real turbofan aircraft engine telemetry using a hybrid Machine Learning + Deep Learning approach:

Isolation Forest — unsupervised anomaly detection

Autoencoder Neural Network — reconstruction-based anomaly scoring

The dataset used is NASA CMAPSS FD001, containing multivariate sensor readings collected across multiple operational cycles until failure.

This methodology supports aviation safety, defence telemetry, predictive maintenance, and reliability engineering.

📂 Project Structure
anomaly_detection/
│
├── dataset/                 # Place NASA FD001 dataset here after download
│
├── src/
│   ├── preprocessing.py     # Data loading & normalization
│   ├── anomaly_detection.py # Model training & hybrid anomaly detection
│   └── visualize.py         # Plotting & result visualization
│
├── results/
│   ├── anomaly_plot.png     # Final anomaly visualization
│   └── anomaly_log.txt      # Detected anomalies with values
│
├── requirements.txt
└── README.md

🎯 Dataset Download (Important)

The NASA CMAPSS dataset is not included due to size and licensing rules.

Download from:
https://data.nasa.gov/dataset/CMAPSS/

Then place FD001 file here:

dataset/train_FD001.txt

🛠 Technologies Used

Python

NumPy, Pandas

Scikit-learn — Isolation Forest

TensorFlow / Keras — Autoencoder

Matplotlib — Visualization

🚀 How It Works

✅ 1. Load & preprocess data

Read CMAPSS FD001 file

Remove empty columns

Normalize all sensor values

Smooth noisy signals

✅ 2. Train Isolation Forest
Detects anomalies caused by:

sensor spikes

unusual operating conditions

abnormal behavior patterns

✅ 3. Train Autoencoder

Learns normal signal structure

High reconstruction error → anomaly

✅ 4. Hybrid decision logic

If Isolation Forest OR Autoencoder flags anomaly → mark as anomaly


✅ 5. Generate results

visual anomaly plot

anomaly detection log file

▶️ Running the Project

Install dependencies:

pip install -r requirements.txt


Run detection:

python src/anomaly_detection.py

📈 Sample Output

Detected anomalies (red) among normal sensor readings:

🎯 Why This Project Matters (DRDO Relevance)

Uses aviation-grade sensor telemetry

Demonstrates predictive maintenance

Implements hybrid ML + DL fault detection

Applicable to aircraft engines, UAV propulsion, missile systems

Reflects real analytical workflows used in defence labs

📚 Future Enhancements

LSTM / GRU sequence-based anomaly detection

Sensor correlation & root-cause analysis

Real-time streaming anomaly monitoring

Explainability using SHAP / feature attribution

📜 License

MIT License — free to use and modify.

👤 Author

Sravani Teeda
CSE (AI & ML), 2026
Open to Research & Defence Internships