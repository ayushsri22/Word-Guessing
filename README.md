# Word-Guessing

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Model](#model)
- [Usage](#usage)
- [Results](#results)

## Introduction
This document outlines the design decisions and calculations behind the development of a machine learning algorithm using a Decision Tree Classifier for bigram-based word prediction. The task involves processing textual data where each word is represented by a sequence of lexicographically sorted bigrams. This study addresses the challenge of efficiently predicting words using only the first five unique bigrams of each word. The bigrams are extracted in a manner that ensures lexicographical order and removes duplicates, facilitating a streamlined input for prediction models. By leveraging these preprocessed bigrams, the prediction system aims to accurately suggest potential words associated with a provided set of bigrams, optimizing both computational efficiency and predictive accuracy in natural language processing applications.

## Project Overview
The primary objective of this project is to predict words based on given bigrams as input, where bigrams are lexicographically sorted and limited to the first five unique bigrams per word. The challenge is predicting words that share the same set of bigrams after preprocessing. 

The Decision Tree Classifier was chosen due to its simplicity and interpretability. Decision trees are non-parametric models that work well with categorical data and can handle complex decision boundaries expected in text classification tasks such as bigram-based word prediction.

## Model

### Approach
1. **Bigram Calculation**: The function `extract_bigrams(term)` generates bigrams from each word. Bigrams are sequences of two consecutive characters from a word, capturing local structure and context within words.
2. **Preprocessing and Feature Extraction**: The `organize_words(terms)` function organizes words by their bigrams and collects all unique bigrams. It limits the number of bigrams to the first five unique sorted bigrams for each word.
3. **Label Encoding**: `LabelEncoder` converts target words into numeric labels, as the decision tree implementation in scikit-learn requires numeric input for the target variable.
4. **MultiLabel Binarization**: `MultiLabelBinarizer` transforms the set of bigrams (features) into a binary matrix suitable for the decision tree algorithm.
5. **Model Training**: The `train` method of the `SequenceModel` class fits the `DecisionTreeClassifier` on the transformed bigram features and the encoded word labels.
    - **Criterion**: `DecisionTreeClassifier(criterion='entropy')` uses entropy (information gain) as the criterion for splitting nodes.
    - **Stopping Criterion and Pruning**: Default settings of `DecisionTreeClassifier` are used, meaning the tree grows until all leaves are pure or contain fewer than the minimum samples required to split.
6. **Prediction**: The `predict` method transforms input bigrams into the same binary matrix format using `MultiLabelBinarizer` and predicts the word labels using the trained decision tree model. The predicted numeric labels are then inverse transformed to obtain the original word labels.

### Summary of Hyperparameters and Design Choices
- **Bigrams Limitation**: Each word is limited to the first five unique bigrams.
- **Entropy Criterion**: Used for selecting the best splits in the decision tree.
- **No Explicit Pruning**: Default settings are used; however, pruning can be added if needed to manage overfitting.

## Usage

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/cs771-assignment2.git
    cd cs771-assignment2
    ```
2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Model
1. Prepare the dataset: Ensure that your dataset is in the `data` directory.
2. Train the model:
    ```bash
    python train.py
    ```
3. Make Predictions:
    Use the provided script to make predictions on new data:
    ```bash
    python predict.py --input data/new_data.csv --output results/predictions.csv
    ```
    **Parameters**:
    - `--input`: Path to the new data file.
    - `--output`: Path to save the prediction results.

## Results
The implemented word prediction model, utilizing lexicographically sorted bigrams with duplicates removed and limited to the first five bigrams, achieved notable performance metrics. The model demonstrated a high precision score of 0.97, indicating its strong accuracy in predicting the correct words based on the given bigrams. Furthermore, the training process was highly efficient, with a training time of only 0.0097 seconds, underscoring the model’s computational efficiency. These results highlight the effectiveness and practicality of the proposed approach in word prediction tasks.

