# 🧠 Intelligent Anomaly Detection in Network Traffic
This project detects anomalous patterns in network traffic using machine learning techniques, with the goal of identifying potential threats such as intrusions and unusual flows in real-time.

## 🚀 Features
- 📥 Ingests network flow data (e.g., from Zeek, pcap, NetFlow)
- 🧮 Extracts statistical features from flows or sessions
- 🤖 Applies machine learning models (e.g., Isolation Forest, Autoencoder) to detect anomalies
- 🔔 Flags high-risk traffic for alerting or review or even block network traffic
- 📊 (Optional) Web dashboard for real-time anomaly monitoring and feedback

## 📈 How It Works
- Capture or load network traffic data (from dataset).
- Preprocess and extract features like bytes, packets, durations, etc.
- Normalize data to ensure consistent scale.
- Run anomaly detection models (e.g., Isolation Forest, Autoencoder) on feature data.
- Score traffic and flag anything above the threshold as an anomaly.
- (Optional) Display results in a web dashboard or write to logs for review.

## 📂 Project Structure
├── data/               # Raw and preprocessed flow data

├── models/             # Trained ML models

├── src/

│   ├── ingest.py       # Network data parsing & loading

│   ├── features.py     # Feature extraction

│   ├── detect.py       # Anomaly detection logic

│   └── train.py        # Model training script

├── dashboard/          # Frontend code (optional)

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
clone this repo

download a dataset of your choice

drop the dataset under the samples folder