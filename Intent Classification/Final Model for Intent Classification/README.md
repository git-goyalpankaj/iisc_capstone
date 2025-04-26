# Final model for Intent Classification

## Overview

The following files have been uploaded for the intent classification task using a fine-tuned BERT model:

- **`Best_Bert_Model.zip`- [Link](https://drive.google.com/file/d/1iZtbG3COTwfGvYCemXSI9kET6kVItXu1/view?usp=sharing)**  
  This file contains the final trained BERT model along with its tokenizer. The model was fine-tuned on text data to perform intent classification based on BERT embeddings.

- **`label_encoder.pkl` - [Link](https://drive.google.com/file/d/1m_d6GL5bfgzqfSCrN7rZdrjGw42nNR2y/view?usp=sharing)**  
  This file contains the `LabelEncoder` object that was fitted during the training phase. It is used to map model prediction indices back to the original intent label names.

- **Usage Script - [Link](https://colab.research.google.com/drive/1jMfj5IExddIxh5kefabpLghMn5SQxt42?usp=sharing)**  
  A script has also been provided that loads the BERT model, tokenizer, and label encoder, and allows you to make predictions on new input queries.

## Purpose

These files together allow you to **reuse the trained model** without retraining. You can directly **load the model and label encoder**, and **predict the intent** for any new user query.

## How It Fits Into the Main Code

- First, unzip the BERT model.
- Load both the model and tokenizer from the extracted folder.
- Load the `label_encoder.pkl` to correctly decode prediction outputs.
- Use the provided script to pass a text query and receive the predicted intent label.

This setup ensures that the model, tokenizer, and label mappings remain **consistent with the original training process**, enabling accurate and reliable predictions.
