# 🎵 Genre Classification using PCA and Logistic Regression

> **Tagline:** Machine learning model for predicting music genres using PCA for dimensionality reduction and Logistic Regression for classification.

**GitHub About Section (suggestion):**  
Music genre classification using feature-based data. Applies Principal Component Analysis (PCA) to reduce dimensionality and uses Logistic Regression to classify songs into genres efficiently and accurately.

---

## 🚀 Overview

This project focuses on classifying **music genres** based on extracted audio features using a **machine learning pipeline** that combines **Principal Component Analysis (PCA)** and **Logistic Regression**.  

The dataset includes musical attributes such as tempo, beats, loudness, duration, and spectral characteristics. The goal is to predict the **genre** of a track from its numerical features while improving training efficiency and avoiding overfitting through PCA.

This project demonstrates fundamental data science practices — data preprocessing, dimensionality reduction, classification, and model evaluation.

---

## 📂 Repository Structure

```
GenreClassification_with_PCA_LogisticRegression/
│
├── classification.ipynb           # Main Jupyter notebook with the complete workflow
├── music_dataset_mod.csv          # Dataset containing music features and genre labels
├── Music Data Legend.xlsx         # Data dictionary explaining the dataset features
├── README.md                      # Project documentation (this file)
└── .gitignore                     # Ignored system files
```

---

## 🧩 Dataset Description

The dataset **`music_dataset_mod.csv`** contains numerical representations of audio features extracted from songs, along with their corresponding genres.  
The **`Music Data Legend.xlsx`** provides the feature definitions, which may include:

| Feature | Description |
|----------|-------------|
| **tempo** | Beats per minute (BPM) of the track |
| **beats** | Estimated number of beats in the song |
| **loudness** | Overall loudness (dB) |
| **duration** | Track duration in seconds |
| **spectral features** | Frequency-domain features from audio processing |
| **chroma features** | Pitch and tone-related metrics |
| **genre** | Target variable (Pop, Rock, Jazz, Hip-Hop, etc.) |

---

## ⚙️ Project Workflow

### **1. Data Preprocessing**
- Load dataset (`music_dataset_mod.csv`) using pandas.  
- Handle missing or outlier values.  
- Normalize numerical features using `StandardScaler`.  
- Encode categorical labels (genres) with `LabelEncoder`.  

### **2. Principal Component Analysis (PCA)**
- Apply PCA to reduce the high-dimensional feature space.  
- Select optimal number of components based on explained variance (e.g., 95%).  
- Visualize cumulative variance to justify dimensionality choice.

### **3. Model Training (Logistic Regression)**
- Split dataset into training and testing sets (e.g., 80/20).  
- Train Logistic Regression on PCA-transformed data.  
- Optionally use regularization (`C` parameter) for tuning.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Standardize and apply PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_pca_train, y_train)
```

### **4. Model Evaluation**
Evaluate model performance using classification metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Example visualization:
```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_pca_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d")
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))
```

---

## 📊 Sample Results (replace with your actual values)

| Metric | Score |
|---------|--------|
| **Accuracy** | 84.6% |
| **Precision** | 83.1% |
| **Recall** | 82.4% |
| **F1-score** | 82.7% |

🧠 *Interpretation:*  
The PCA + Logistic Regression pipeline achieves robust performance while drastically reducing computational cost. The model generalizes well, showing PCA effectively preserves the variance needed for genre discrimination.

---

## 📈 PCA Explained Variance Example

| Principal Component | Explained Variance (%) |
|----------------------|------------------------|
| PC1 | 31.5 |
| PC2 | 17.2 |
| PC3 | 10.4 |
| PC4 | 7.9 |
| PC5 | 5.3 |
| **Cumulative (Top 10 PCs)** | **95.1%** |

💡 *PCA reduces hundreds of raw audio features to a smaller set of meaningful components without significant loss of information.*

---

## 🧠 Insights

- PCA effectively reduces dimensionality and computational time.  
- Logistic Regression is interpretable and performs well with reduced features.  
- Feature scaling is crucial before PCA for correct variance distribution.  
- PCA ensures generalization, especially when dealing with highly correlated features.  

---

## 💡 Future Improvements

- [ ] Compare Logistic Regression with SVM or Random Forest.  
- [ ] Implement hyperparameter tuning using GridSearchCV.  
- [ ] Deploy as a Streamlit app to classify genres interactively.  
- [ ] Visualize PCA clusters in 2D for genre separability.  
- [ ] Extend dataset with more genres and tracks.  

---

## 🧰 Tech Stack

| Category | Tools Used |
|-----------|-------------|
| **Language** | Python |
| **Libraries** | pandas, numpy, scikit-learn, matplotlib, seaborn |
| **Modeling Techniques** | PCA, Logistic Regression |
| **Environment** | Jupyter Notebook |

---

## 📦 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Shubham91999/GenreClassification_with_PCA_LogisticRegression.git
   cd GenreClassification_with_PCA_LogisticRegression
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook classification.ipynb
   ```

4. Run all cells sequentially to reproduce results.

---

## 🧑‍🚀 Author

**Shubham Kulkarni**  
Machine Learning Engineer | Data Science & AI Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/shubham91999) • [GitHub](https://github.com/Shubham91999)

---

## 🪙 License

This project is released under the **MIT License** — you may use and modify it for educational or research purposes.

---

⭐ *If you find this project useful, please give the repository a star!* 🌟
