import pandas as pd

def load_cmapss(path):
    # Read the file while automatically removing empty columns
    df = pd.read_csv(path, sep=r"\s+", header=None)
    
    # Drop extra empty columns (common in NASA data)
    df = df.dropna(axis=1, how='all')

    # Now rename columns dynamically
    # First column = engine id
    # Second column = cycle
    # Remaining = sensors
    col_count = df.shape[1]

    # Minimum: 26 columns, Maximum: 28 columns
    # Dynamic handling makes it robust
    col_names = ['unit', 'cycle'] + [f'sensor_{i}' for i in range(1, col_count - 1)]

    df.columns = col_names

    # Normalize only sensor columns
    sensor_cols = [c for c in df.columns if 'sensor_' in c]

    df[sensor_cols] = (df[sensor_cols] - df[sensor_cols].mean()) / df[sensor_cols].std()

    return df, sensor_cols
