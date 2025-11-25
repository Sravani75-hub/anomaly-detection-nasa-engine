📘 AI-Based Anomaly Detection on NASA Turbofan Engine Sensor Data (CMAPSS)
📌 Overview

This project identifies abnormal behavior in real turbofan aircraft engine sensor data using a hybrid machine-learning approach:

Isolation Forest — unsupervised anomaly detection

Autoencoder Neural Network — reconstruction-based anomaly detection

The dataset used is NASA CMAPSS FD001, which contains multivariate engine sensor recordings over multiple operational cycles until failure.

This methodology is widely used in aviation, defence telemetry, predictive maintenance, and reliability engineering.

📂 Project Structure
anomaly_detection/
│
├── dataset/                 # Place NASA FD001 dataset here after download
│
├── src/
│   ├── preprocessing.py     # Data loading & normalization
│   ├── anomaly_detection.py # Model training & anomaly detection pipeline
│   └── visualize.py         # Plots & visual analysis
│
├── results/
│   ├── anomaly_plot.png     # Visual anomaly representation
│   └── anomaly_log.txt      # Full anomaly detection output
│
├── requirements.txt
└── README.md

🎯 Dataset Download (Important)

The NASA CMAPSS FD001 dataset is not included due to size and licensing restrictions.

Download it from:

https://data.nasa.gov/dataset/CMAPSS/

Then place the file inside:

dataset/train_FD001.txt

🛠 Technologies Used

Python

NumPy, Pandas

Scikit-learn (Isolation Forest)

TensorFlow / Keras (Autoencoder)

Matplotlib (visualization)

🚀 How It Works
✅ 1. Load & Preprocess Data

Read CMAPSS FD001 sensor data

Remove empty columns

Normalize all sensor channels

Smooth noisy signals

✅ 2. Train Isolation Forest

Detects statistical outliers caused by:

sudden sensor spikes

unusual operating conditions

abnormal patterns

✅ 3. Train Autoencoder Neural Network

Learns normal engine behavior and reconstruction patterns.
Higher reconstruction error → anomaly.

✅ 4. Combine Model Decisions

Final anomaly = flagged by either model → reduces false negatives.

✅ 5. Generate Outputs

visual anomaly plot

anomaly detection log

▶️ Running the Project
pip install -r requirements.txt
python src/anomaly_detection.py

📈 Sample Output

Below is a visualization showing detected anomalies (red points)
among normal engine sensor readings:

🎯 Why This Project Matters (DRDO Relevance)

Uses real aviation-grade sensor telemetry

Demonstrates predictive maintenance capabilities

Shows hybrid ML + DL anomaly detection approach

Applicable to aircraft, UAVs, missile engines & ground test systems

Reflects real research workflows used in defence labs

📚 Future Enhancements

LSTM / GRU sequence-based anomaly detection

Sensor correlation analysis for root cause detection

Real-time streaming anomaly detection pipeline

Model explainability (SHAP / feature attribution)

📜 License

This project is open-source under the MIT License.

👤 Author

Sravani Teeda
CSE (AI & ML), 2026
Open to Research & Defence Internships