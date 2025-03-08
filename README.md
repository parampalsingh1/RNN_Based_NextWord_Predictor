# 🚀 RNN-Based Next-Word Prediction Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)  
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)

---

## 🔍 **Overview**

This project implements an **RNN-based next-word prediction model** using **TensorFlow/Keras**. The model is designed to predict the next word in a sentence by training on a given text dataset. The process includes comprehensive **data preprocessing**, which involves:
- Converting text to lowercase
- Removing punctuation
- Tokenization
- Creating input-output sequences using a sliding-window approach

Once trained, the model generates next-word predictions based on user input.

**Technologies & Libraries:**
- **Python 3** (with libraries such as **NumPy**, **Pandas**, **Matplotlib**)
- **TensorFlow/Keras** for deep learning model building
- **Matplotlib** for visualizing model training performance

---

## ✨ **Features**

- **Advanced Text Preprocessing:**  
  - Converts the dataset into lowercase
  - Removes unnecessary punctuation
  - Tokenizes the text and creates sequences for model training

- **Sliding-Window Tokenization & Sequencing:**  
  - Converts the dataset into numerical sequences using a sliding window, ensuring that the model learns the relationships between sequential words

- **Deep Learning Model Architecture:**  
  - Uses an **Embedding** layer followed by **SimpleRNN** layers to capture temporal dependencies and predict the next word
  - The final layer is a **Dense** layer with a **softmax** activation function for classification

- **Training Insights:**  
  - Tracks model accuracy and loss during training
  - Plots graphs to visualize training performance over epochs (accuracy vs. loss)

- **Interactive Prediction:**  
  - The model accepts user input in the form of a sentence and predicts the next word based on its training

---

## 🛠️ **Setup & Prerequisites**

- Python **3.8** or higher
- **TensorFlow 2.0+** (or later)
- Other required libraries: `numpy`, `matplotlib`, `keras`, `pandas`

