# Industrial Safety Chatbot

A machine learning and deep learning system that classifies industrial accident incidents by severity level, enabling automated risk assessment from accident descriptions and metadata.

---

## Project Overview

This capstone project builds a multi-class classification pipeline to predict the **accident severity level** (5 classes) from structured incident data and free-text accident descriptions. The system is designed to support real-time industrial safety monitoring by automatically triaging reported incidents.

---

## Dataset

| Property | Details |
| --- | --- |
| Total Records | 425 industrial accident incidents |
| Time Period | 2016–2017 |
| Missing Values | None (100% complete) |
| Vocabulary Size | 15,055 unique words (post-processing) |

### Features

| Column | Description |
| --- | --- |
| Date | Incident date and time |
| Countries | Country where incident occurred |
| Local | Specific location |
| Industry Sector | Type of industry |
| Accident Level | **Target variable** — severity (5 classes: 0–4) |
| Potential Accident Level | Risk level classification |
| Genre | Type of incident |
| Employee or Third Party | Who was affected |
| Critical Risk | Risk category |
| Description | Free-text description of the accident |

---

## Project Structure

```text
Industrial_Safety_Chatbot/
├── Industrial_Chatbot_Capstone_EDA-1.ipynb                              # Exploratory Data Analysis
├── Industrial_Chatbot_Capstone_Modelling_Machine_Learning_Sampling_Balanced-1.ipynb   # ML Models
├── Industrial_Chatbot_Capstone_Modelling_Deep_Learning_Sampling_Balanced-1.ipynb     # Deep Learning Model
├── Industrial Safety Chatbot - GUI-1.zip                                # Chatbot GUI
├── Industrial Safety Chatbot - Final Report (2).pdf                     # Full project report
└── Model Architecture-1.png                                             # DL model architecture diagram
```

---

## Methodology

### 1. Exploratory Data Analysis

- Distribution analysis across years and months (peak: April with 61 incidents)
- N-gram analysis (unigrams, bigrams, trigrams) on accident descriptions
- Correlation analysis between numerical features
- Class imbalance investigation across 5 severity levels

### 2. Text Preprocessing

- Lowercasing, URL/emoji removal, contraction expansion, punctuation removal
- Tokenization, stopword removal, stemming, and lemmatization
- GloVe 200-dimensional pre-trained word embeddings (400K word vocabulary)

### 3. Handling Class Imbalance

- Random under-sampling
- SMOTE (Synthetic Minority Oversampling Technique)
- Train-test split: 75/25 (345 training, 115 test samples after sampling)

### 4. Feature Encoding

- Label encoding, Ordinal encoding, Binary encoding for categorical features
- StandardScaler for ordinal-encoded numerical features

---

## Models

### Machine Learning

| Model | Accuracy |
| --- | --- |
| Logistic Regression | 85% |
| Gaussian Naive Bayes | 85% |
| Decision Tree | 84% |
| Gradient Boosting | 84% |
| Random Forest | 84–92% |
| Bagging Classifier | 80% |
| SVM (poly, C=1000) | 81% |
| K-Nearest Neighbors | 79% |
| Stacking Classifier | 88–89% |
| **Voting Classifier** | **91%** |

> Best ML model: **Voting Classifier** with 91% test accuracy, 92% precision, 91% recall, 90% F1-score (weighted).

### Deep Learning

Dual-input neural network combining text and metadata streams:

- **Text Branch**: GloVe embedding (200-dim) → Bidirectional LSTM (200→100 units) → TimeDistributed Dense → Flatten
- **Metadata Branch**: Dense layers (1024 → 512 → 256) with BatchNorm, ReLU, and Dropout (0.1–0.3)
- **Fusion**: Concatenate both streams → Dense layers (64 → 32 → 16 → 5) with Softmax output
- **Total Parameters**: ~15.87M
- **Optimizer**: Adam | **Loss**: Categorical Crossentropy
- **Callbacks**: Early Stopping, Learning Rate Reduction

![Model Architecture](Model%20Architecture-1.png)

---

## Results Summary

| Approach | Best Model | Test Accuracy |
| --- | --- | --- |
| Machine Learning | Voting Classifier | 91% |
| Deep Learning | Bidirectional LSTM + Dense Fusion | Demonstrated strong multi-class discrimination |

---

## Tech Stack

- **Language**: Python
- **ML Libraries**: Scikit-learn, XGBoost
- **DL Framework**: TensorFlow / Keras
- **NLP**: NLTK, GloVe Embeddings
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Imbalance Handling**: imbalanced-learn (SMOTE)

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Drishti-Chopra/Industrial_Safety_Chatbot.git
   cd Industrial_Safety_Chatbot
   ```

2. Install dependencies:

   ```bash
   pip install pandas numpy scikit-learn tensorflow keras nltk xgboost imbalanced-learn matplotlib seaborn
   ```

3. Run notebooks in order:
   - `Industrial_Chatbot_Capstone_EDA-1.ipynb`
   - `Industrial_Chatbot_Capstone_Modelling_Machine_Learning_Sampling_Balanced-1.ipynb`
   - `Industrial_Chatbot_Capstone_Modelling_Deep_Learning_Sampling_Balanced-1.ipynb`

4. For the chatbot GUI, extract `Industrial Safety Chatbot - GUI-1.zip` and follow instructions inside.

---

## Author

**Drishti Chopra**  
[GitHub](https://github.com/Drishti-Chopra)
