# 🌍 Earthquake Prediction using CatBoost

This project leverages machine learning techniques, particularly **CatBoost**, to predict **time to failure** in earthquake occurrences based on acoustic data. The dataset is processed and features are engineered for effective model training.


![CatBoost Logo]([https://upload.wikimedia.org/wikipedia/commons/3/36/Catboost-logo.png](https://www.google.com/url?sa=i&url=https%3A%2F%2Fdatascientest.com%2Fen%2Fcatboost-an-essential-machine-learning-tool&psig=AOvVaw1KK1KqqvagfQExELKmF_g2&ust=1726993196399000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCOj1kpHN04gDFQAAAAAdAAAAABAE))

## 📁 Project Structure

```bash
├── data/
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset
├── Earthquake Prediction.ipynb  # Jupyter Notebook containing model code
├── README.md                    # Project documentation
└── requirements.txt             # Dependencies
```

## 📊 Dataset

The data is a time series with acoustic_data as the input and time_to_failure as the target.

Source: Kaggle Earthquake Challenge

### 🔧 Installation & Dependencies

To set up the environment, install the required dependencies using:

```
pip install -r requirements.txt
```

Main libraries used:

CatBoost

Pandas

NumPy

Matplotlib

Scikit-learn

## 🚀 How to Run
Preprocess Data:

Feature engineering is done using a custom function gen_features() to extract useful patterns from acoustic_data.

Train the Model:

The CatBoostRegressor is used for predicting the time_to_failure with the MAE loss function and ordered boosting.
```
train_pool = Pool(X_train, y_train)
model = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')
model.fit(train_pool, silent=True)
```

Model Evaluation:

Once the model is trained, predictions are made on the test data and evaluated using suitable metrics.

## ⚠️ Common Errors & Fixes
CatBoostError: Length of label and length of data is different

Ensure that the number of rows in X_train matches the number of labels in y_train.

Use the following code to append the correct number of labels during training:

python
```
for df in train:
    ch = gen_features(df['acoustic_data'])
    X_train = pd.concat([X_train, ch], ignore_index=True)
    num_rows = len(ch)
    label = pd.Series([df['time_to_failure'].values[-1]] * num_rows)
    y_train = pd.concat([y_train, label], ignore_index=True)
```

## 🛠️ Feature Engineering
The gen_features() function extracts essential features such as:

Mean and Standard Deviation of acoustic data

Rolling statistics to capture trends over time

FFT transforms for frequency domain analysis

## 🔮 Future Enhancements

🔍 Additional Models: Experiment with other machine learning models like XGBoost or LightGBM.

🧠 Deep Learning: Investigate the use of LSTM or GRU for time series data.

📈 Hyperparameter Tuning: Use GridSearchCV to find the optimal hyperparameters for better model performance.

## 📊 Visualizations
Explore trends in the data through visualizations:

```
import matplotlib.pyplot as plt

plt.plot(df['acoustic_data'])
plt.title("Acoustic Data over Time")
plt.show()
```
## 🤝 Contributing
Feel free to contribute to the project by submitting a pull request. Contributions can be feature additions, bug fixes, or performance improvements.

## 📜 License
This project is licensed under the MIT License.

## 👨‍💻 Author
Shaikh Asif Hossain






































