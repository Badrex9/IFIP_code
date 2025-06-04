# Data Transformation for IDS: Leveraging Symbolic and Temporal Aspects

This repository contains the code and experiments associated with the publication:

**Data Transformation for IDS: Leveraging Symbolic and Temporal Aspects**  
🔗 DOI: [10.1007/978-3-031-92882-6_14](https://doi.org/10.1007/978-3-031-92882-6_14)

---

## 🧠 Overview

This work proposes a novel approach to enhance Intrusion Detection Systems (IDS) by applying **symbolic** and **temporal** transformations to raw network traffic data. The goal is to make patterns more discriminative and interpretable for downstream machine learning models.


## 📁 File Overview

- `creation_SW_ip_train.py` – Preprocesses raw training data to generate symbolic and temporal representations.
- `creation_SW_ip_test.py` – Applies the same transformation to test data.
- `entrainement.py` – Trains our CNN model.
- `predict.py` – Loads the trained model and performs predictions on test data.
- `test.ipynb` – Jupyter notebook to visualize, test and analyze the results.
