import matplotlib.pyplot as plt

def plot_results(df, sensor_cols):
    # Choose one sensor to visualize (sensor_2 is common)
    sensor = sensor_cols[1]

    normal = df[df['anomaly'] == 0]
    anom = df[df['anomaly'] == 1]

    plt.figure(figsize=(14,5))
    plt.plot(normal[sensor].values, label="Normal", linewidth=0.8)
    plt.scatter(anom.index, anom[sensor].values, color='red', s=10, label="Anomaly")

    plt.title("NASA Turbofan Engine – Sensor Data Anomaly Detection")
    plt.xlabel("Time")
    plt.ylabel("Sensor Value (Normalized)")
    plt.legend()

    plt.savefig("results/anomaly_plot.png", dpi=300)
    plt.close()
