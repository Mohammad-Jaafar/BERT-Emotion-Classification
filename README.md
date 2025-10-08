# Emotion Classification with BERT

## Overview

This project demonstrates how to build, train, and evaluate a
**transformer-based deep learning model** for **emotion classification**
using the [Emotion
Dataset](https://huggingface.co/datasets/dair-ai/emotion).\
The model leverages **BERT (Bidirectional Encoder Representations from
Transformers)** to classify text into six emotional categories:
**sadness**, **joy**, **love**, **anger**, **fear**, and **surprise**.

------------------------------------------------------------------------

## Features

-   Automatic download and preprocessing of the Emotion dataset using
    **Hugging Face Datasets**.
-   Tokenization and truncation using a pre-trained **BERT tokenizer**.
-   Training with the **Hugging Face Trainer API**.
-   Evaluation with **classification_report** for detailed performance
    metrics.
-   Model and tokenizer saving for later inference.

------------------------------------------------------------------------

## Technologies Used

-   **Python 3.9+**
-   **Transformers (Hugging Face)**
-   **Datasets**
-   **PyTorch**
-   **Scikit-learn**

------------------------------------------------------------------------

## Project Structure

    Emotion-Classification-BERT
    │── BERT_Emotion_Classification_TrainCode.ipynb
    │── requirements.txt
    │── README.md

------------------------------------------------------------------------

## Installation & Usage

1.  **Clone the repository:**

    ``` bash
    git clone https://github.com/Mohammad-Jaafar/emotion-classification-bert.git
    cd emotion-classification-bert
    ```

2.  **Install dependencies:**

    ``` bash
    pip install transformers datasets torch scikit-learn
    ```

3.  **Run the script:**

    ``` bash
    python emotion_classification_bert.ipynb
    ```

4.  The trained model and tokenizer will be saved in the
    `sentiment-bert/` directory.

------------------------------------------------------------------------

## Results

After training for **3 epochs**, the model achieved the following
results on the test set:

**Overall Accuracy:** \~93%\
**Training Loss Progress (Epoch 3/3):**

    Step   Training Loss
    500    0.7568
    1000   0.2384
    1500   0.1408
    2000   0.1393
    2500   0.1014
    3000   0.0907

------------------------------------------------------------------------

## Demo on HuggingFace Spaces
- **BERT Emotion Classification**  
[HuggingFace](https://huggingface.co/spaces/Mhdjaafar/BERT-Emotion-Classification)

------------------------------------------------------------------------
## Future Work

-   Fine-tune a **DistilBERT** or **RoBERTa** model for comparison.
-   Add support for **multilingual** emotion datasets.
-   Experiment with **data augmentation** and **hyperparameter tuning**.

------------------------------------------------------------------------

## Author

-   **Mohammad Jaafar**\
    [LinkedIn](https://www.linkedin.com/in/mohammad-jaafar-) \|
    [HuggingFace](https://huggingface.co/Mhdjaafar)

------------------------------------------------------------------------

## License

This project is licensed under the **MIT License**.\
Feel free to use, modify, and distribute with attribution.
