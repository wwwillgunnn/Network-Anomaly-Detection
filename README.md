# 🧠 Intelligent Anomaly Detection in Network Traffic
This project detects anomalous patterns in network traffic using machine learning techniques, with the goal of identifying potential threats such as intrusions and unusual flows in real-time.

## 🚀 Features
- 📥 Ingests network flow data
- 🧮 Extracts statistical features from data
- 🤖 Applies machine learning models (e.g., Isolation Forest, Autoencoder, LSTM) to detect anomalies
- 🔔 Flags high-risk traffic for alerting or review
- 📊 Web dashboard for real-time anomaly monitoring and feedback

## 📈 How It Works
- Load network traffic data from dataset.
- Preprocess and extract features like bytes, packets, durations, etc.
- Normalize data to ensure consistent scale.
- Run anomaly detection models (e.g., Isolation Forest, Autoencoder) on feature data.
- Score traffic and flag anything above the threshold as an anomaly.
- (Optional) Display results in a web dashboard or write to logs for review.

## 📂 Project Structure
    ├── data/               # Raw and preprocessed flow data 
   
        ├── processed/      # processed datasets from ingest.py
        
        ├── samples/        # raw csv files from kaggle

    ├── models/             # Trained ML models

    ├── src/

        ├── utils/         # Folder full of helper functions

        ├── ingest.py       # Network data parsing & loading

        ├── features.py     # Feature extraction

        ├── detect.py       # Anomaly detection logic

        └── train.py        # Model training script

    ├── dashboard/          # Frontend code

    ├── README.md

    └── requirements.txt

## 📊 Sample Output
Timestamp	Src IP	Dst IP	Protocol	Score	Anomaly

2025-08-05 14:00	10.0.0.2	192.168.1.5	TCP	0.97	✅ Yes

2025-08-05 14:01	10.0.0.3	192.168.1.10	UDP	0.23	❌ No

## 📚 Datasets
- CIC-IDS2017: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset
- CIC-IDS2018: https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv
- CIC-IDS2019: https://www.kaggle.com/datasets/tarundhamor/cicids-2019-dataset
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Set up
### 1) Clone the repo and install dependencies
```bash
git clone <YOUR_REPO_URL>
cd Network-Anomaly-Detection
pip install -r requirements.txt
```

### 2) Download a dataset of your choice
download and drop the dataset of your choice under data/samples

### 3) Generate a clean dataset
```bash
python ingest.py
```

### 4) Train models
```bash
python train.py
```
### 5) Launch dashboard
```bash
cd dashboard
streamlit run app.py
```