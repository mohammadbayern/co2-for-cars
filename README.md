# 🚗 Vehicle CO₂ Emissions Prediction using Deep Learning

## 📌 Project Overview

This project predicts **vehicle CO₂ emissions** using a **Deep Learning regression model** built with TensorFlow/Keras.

The project demonstrates an end-to-end machine learning workflow, including:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature scaling
* Prevention of data leakage using Scikit-learn Pipeline
* Neural network modeling
* Model evaluation and visualization

The objective is to learn the complex non-linear relationship between vehicle characteristics and their corresponding CO₂ emissions.

This project is suitable for:

* Master's degree applications
* Machine Learning & Deep Learning portfolios
* Data Science interviews
* Environmental data analysis projects

---

# 📊 Dataset

The dataset contains numerical vehicle characteristics used to predict CO₂ emissions.

### Target Variable

* **out1** → Vehicle CO₂ Emissions (g/km)

### Input Features

The model uses five numerical features describing vehicle specifications and fuel consumption.

---

# 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed before model development:

* Dataset inspection
* Missing value verification
* Statistical summary
* Target distribution analysis
* Correlation heatmap
* Feature relationship visualization

These analyses helped verify that the dataset was suitable for a regression task.

---

# ⚙️ Data Preprocessing

To ensure reliable model performance and eliminate data leakage, the preprocessing pipeline follows these steps:

1. Train-test split (80/20)
2. Feature standardization using **StandardScaler**
3. Scaling performed through **Scikit-learn Pipeline**, where the scaler is fitted only on the training data and then applied to the test set.

This workflow follows machine learning best practices and prevents information leakage from the test data.

---

# 🧠 Deep Learning Model

The regression model was implemented using TensorFlow/Keras.

Architecture:

* Input layer
* Hidden layer (55 neurons, ReLU activation)
* Output layer (1 neuron)

Training configuration:

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)
* Epochs: 500

The architecture provides a good balance between simplicity and predictive performance.

---

# 🧩 Neural Network Visualization

The model architecture was visualized using the **keras_visualizer** package.

The generated diagram illustrates:

* Input layer
* Hidden layer
* Output layer
* Layer connectivity

This visualization improves model interpretability and documentation quality.

---

# 📈 Model Training & Evaluation

Model performance was evaluated using:

* Training Loss (MSE)
* Loss convergence curve
* Predicted vs Actual comparison

The trained model successfully captured the non-linear relationship between vehicle features and CO₂ emissions.

---

# 📌 Results

The model demonstrates:

* Stable learning during training
* Successful convergence of the loss function
* Strong regression capability on unseen samples
* Effective application of neural networks to structured tabular data

---

# 🛠️ Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* TensorFlow / Keras
* Scikit-learn
* StandardScaler
* Pipeline
* Jupyter Notebook

---

# 🎯 Project Highlights

This project demonstrates practical knowledge of:

* Exploratory Data Analysis (EDA)
* Data preprocessing
* Feature scaling
* Data leakage prevention
* Deep Learning regression
* Neural network design
* Model visualization
* Machine Learning best practices

---

# 🚀 Future Improvements

Possible future enhancements include:

* Early Stopping
* Validation dataset
* Hyperparameter tuning
* Cross-validation
* Comparison with traditional Machine Learning regression models
* Additional evaluation metrics (MAE, RMSE, R²)
* Model deployment using FastAPI or Streamlit

---

# 👤 Author

**Mohamad Nahvi**

Aspiring Data Scientist | Machine Learning & Deep Learning Enthusiast
