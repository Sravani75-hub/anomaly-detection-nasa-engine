# 🔍 AI-Based Anomaly Detection on NASA Turbofan Engine Sensor Data (CMAPSS)

## 📌 Overview
This project detects abnormal behavior in real aircraft engine sensor data using a combination of:
- **Isolation Forest** (unsupervised ML)
- **Autoencoder Neural Network** (deep learning)

The dataset used is **NASA CMAPSS FD001**, which contains actual turbofan engine sensor readings over multiple cycles until failure.

This is the same type of analytics used in aviation, defence telemetry, and predictive maintenance systems.

---

## 📂 Project Structure
anomaly_detection/
│
├── dataset/
│ └── train_FD001.txt
│
├── src/
│ ├── preprocessing.py
│ ├── anomaly_detection.py
│ └── visualize.py
│
├── results/
│ ├── anomaly_plot.png
│ └── anomaly_log.txt
│
├── README.md
└── requirements.txt

yaml
Copy code

---

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn (Isolation Forest)
- TensorFlow / Keras (Autoencoder)
- Matplotlib (Visualization)

---

## 🚀 How It Works
### **1. Load & Clean Data**
- Reads CMAPSS FD001 dataset  
- Removes blank columns  
- Normalizes sensor values  

### **2. Train Isolation Forest**
Detects statistical outliers based on:
- spikes  
- sudden pattern changes  

### **3. Train Autoencoder**
Learns normal sensor behavior using reconstruction loss.  
Higher error = anomaly.

### **4. Combine Results**
If either model flags an anomaly → final anomaly = 1.

### **5. Output**
✔ `anomaly_plot.png` – graph showing normal (blue) and anomaly (red) points  
✔ `anomaly_log.txt` – full anomaly table with sensor values  

---

## ▶️ Running the Project

pip install -r requirements.txt
python src/anomaly_detection.py

yaml
Copy code

---

## 📈 Sample Output

- A plot showing detected anomalies  
- A detailed log of all anomalous readings  

---

## 🎯 Why This Project is Relevant to DRDO
- Uses real aviation sensor data  
- Demonstrates predictive maintenance capability  
- Shows experience with ML + DL hybrid models  
- Similar to systems used in aircraft, UAV, and missile telemetry monitoring  

---

## 📚 Future Enhancements
- LSTM-based sequence anomaly detection  
- Multi-sensor anomaly scoring  
- Real-time streaming detection 