# 🎓 Student Performance Predictor

An interactive, web-based neural network demo built with **Streamlit** and **TensorFlow/Keras**
(Artificial Neural Network – Activity 3).

## Features
- Synthetic student dataset: study hours, attendance, previous score, sleep hours → PASS / FAIL
- Adjustable dataset size and label noise; 80/20 train–test split
- 1–3 hidden layers, 1–32 neurons, ReLU / tanh / sigmoid, optional dropout
- 5 learning rates (Adam optimizer); train N epochs, single-epoch step, reset
- Live loss and accuracy curves, test confusion matrix, parameter count
- What-if sliders to predict PASS/FAIL for any student
- Permutation feature importance chart

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push this folder to GitHub, then create an app on https://share.streamlit.io pointing to `app.py`.
