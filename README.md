# Deep Learning Fruit Classification Project

This repository contains the implementation and results of two projects focused on classifying fruits using the Fruits-360 dataset. The primary goal of these projects is to explore various deep learning and transfer learning techniques to improve classification accuracy.

## Projects Overview

1. **Fruit Classification with Deep Learning**
    - Classifies fruits using Convolutional Neural Networks (CNNs) and traditional machine learning models.
    - Explores PCA for dimensionality reduction.
    - Implements models like Random Forest, XGBoost, SVM, ANN, and CNN for classification.
    - The project demonstrates a comparison of various models and their performance on the Fruits-360 dataset.

2. **Improving Fruit Classification with Transfer Learning**
    - Utilizes pretrained Convolutional Neural Networks (ConvNets) like ResNet50, MobileNet, and VGG16 for transfer learning and fine-tuning.
    - Achieves improvements in classification accuracy using transfer learning, with insights into model behavior through Grad-CAM visualizations.
    - Includes evaluation metrics such as precision, recall, and F1-score for each class.

## Dataset

The project uses the **Fruits-360** dataset, which contains 141 fruit and vegetable classes with a total of 79,487 images for training and 23,619 images for testing.

- **Dataset Link**: [Fruits-360 Dataset on Kaggle](https://www.kaggle.com/datasets/moltean/fruits/data)
- **Image Dimensions**: 100x100 pixels for traditional models and resized to 224x224 for pretrained models.

## Methodology

### 1. Dimensionality Reduction with PCA
- PCA was applied to reduce the dataset's dimensionality while retaining 80% of the variance. This method was used to transform the input features, improving the efficiency of machine learning models.

### 2. Traditional Machine Learning Models
- The following traditional machine learning models were used to classify fruits:
    - **Random Forest (RF)**: Achieved a test accuracy of **99.71%**, with precision, recall, and F1 score all reaching **99.71%**.
    - **XGBoost (XGB)**: Achieved a test accuracy of **98.14%**, with a precision of **98.16%**.
    - **LightGBM (LGBM)**: Showed low performance with a test accuracy of **14.73%** due to sensitivity in the model configuration.
    - **Support Vector Machine (SVM)**: Achieved a test accuracy of **98.51%** and a precision of **98.55%**.

#### Results for Traditional Machine Learning Models
| Model                 | Test Accuracy | Validation Accuracy | Precision | Recall | F1 Score |
|-----------------------|---------------|---------------------|-----------|--------|----------|
| Random Forest (RF)     | 99.71%        | 99.67%              | 99.71%    | 99.71% | 99.71%   |
| XGBoost (XGB)          | 98.14%        | 97.87%              | 98.16%    | 98.14% | 98.14%   |
| LightGBM (LGBM)        | 14.73%        | 15.10%              | 15.61%    | 14.73% | 13.70%   |
| Support Vector Machine (SVM) | 98.51% | 98.43%              | 98.55%    | 98.51% | 98.50%   |

### 3. Artificial Neural Network (ANN) and Convolutional Neural Networks (CNN)
- An ANN model was developed with **Dense layers** to classify fruits. The architecture was designed with several layers, and **Dropout** layers were added to prevent overfitting.
  
#### ANN Results
- **Test Accuracy**: 75.95%
- **Validation Accuracy**: 77.20%
- **Loss**: 0.82
- **Notes**: ANN showed moderate performance compared to traditional machine learning models, indicating room for optimization in architecture or training.

- A simple CNN model with three **Conv2D layers** followed by **Dense layers** was built for fruit classification.
- The CNN architecture includes layers with 32, 64, and 128 filters, with **MaxPooling2D** used after each Conv2D layer. The model is finalized with a dense layer of 128 neurons followed by the output layer with **softmax activation** for classification.

#### CNN Results
- **Test Accuracy**: 98.96%
- **Validation Accuracy**: 99.12%
- **Loss**: 0.0543
- **Notes**: CNN outperformed the ANN model and rivaled the top-performing machine learning models, showcasing its effectiveness for image classification tasks.

### 4. Transfer Learning
- **ResNet50**, **MobileNet**, and **VGG16** pretrained models were fine-tuned for the classification task. These models were first used for **feature extraction**, followed by **fine-tuning** to further improve performance.
  
  #### **ResNet50**
  - ResNet50 is a deep convolutional neural network with 50 layers. It employs **residual connections**, which help mitigate the vanishing gradient problem and allow deeper architectures to be effectively trained.
  - **Feature Extraction**: Achieved a **test accuracy of 95.00%**.
  - **Fine-Tuning**: After fine-tuning, the accuracy improved to **97.33%**.

  #### **MobileNet**
  - MobileNet is a lightweight network optimized for mobile and edge devices. It uses **depthwise separable convolutions**, which significantly reduce computational cost while maintaining accuracy.
  - **Feature Extraction**: Achieved **92.20% accuracy**.
  - **Fine-Tuning**: Achieved **97.00% accuracy**, showcasing its efficiency for mobile applications.

  #### **VGG16**
  - VGG16 is a widely used ConvNet architecture with 16 layers. It employs **3x3 convolutional layers** with ReLU activation and max-pooling layers.
  - **Feature Extraction**: Achieved **95.77% accuracy**.
  - **Fine-Tuning**: After fine-tuning, the accuracy improved to **99.94%**, achieving excellent results.

#### Training Process
- **Feature Extraction**: The convolutional base of each model was frozen to leverage pretrained weights. Fully connected layers were added for classification.
- **Fine-Tuning**: Specific layers of the convolutional base were unfrozen to enable retraining with a reduced learning rate.
- **Optimization**: **Adam optimizer** with learning rates of **1e-5** and **1e-6** was used during training and fine-tuning.

### 5. Explainability
- **Grad-CAM** and **Grad-CAM++** visualizations were generated to explain the decision-making process of the models. These visualizations highlight the areas of the image that the models focus on when making predictions, improving model interpretability.

## Results and Visualizations

After training, you can visualize the following metrics and results to understand the performance of the models:

### **Accuracy and Loss Plots**
- The training and validation accuracy curves indicate the models' learning progress and their ability to generalize. These plots help assess the effectiveness of each model and the risk of overfitting.
  
  - **Accuracy Plots** show how the model improves with each epoch.
  - **Loss Plots** track the reduction in error during training and validation.

### **Confusion Matrix**
- A confusion matrix is a tool for evaluating classification models. It shows how well the model performs on each fruit class, indicating the true positives, false positives, false negatives, and true negatives.
  - The diagonal elements of the confusion matrix represent the correctly predicted classes, while the off-diagonal elements indicate misclassifications.

### **Grad-CAM Visualizations**
- **Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to visualize which parts of the image the model focuses on when making predictions. It helps interpret the decision-making process of CNNs and other deep learning models.
- Grad-CAM highlights the key regions of the input image that influenced the model's classification decision.
- **Grad-CAM++** provides a more detailed focus, further improving the interpretability of the models by highlighting smaller, more specific regions of interest.

### Summary of Models Performance 

| Model                 | Test Accuracy | Validation Accuracy | Precision | Recall | F1 Score |
|-----------------------|---------------|---------------------|-----------|--------|----------|
| Random Forest (RF)     | 99.71%        | 99.67%              | 99.71%    | 99.71% | 99.71%   |
| XGBoost (XGB)          | 98.14%        | 97.87%              | 98.16%    | 98.14% | 98.14%   |
| LightGBM (LGBM)        | 14.73%        | 15.10%              | 15.61%    | 14.73% | 13.70%   |
| Support Vector Machine (SVM) | 98.51% | 98.43%              | 98.55%    | 98.51% | 98.50%   |


| Model                    | Test Accuracy | Validation Accuracy | Loss  |
|--------------------------|---------------|---------------------|-------|
| Artificial Neural Network (ANN) | 75.95% | 77.20%               | 0.82  |
| CNN (Custom)              | 98.96%        | 99.12%              | 0.0543|
| ResNet50 (Feature Extraction) | 95.0%     | N/A                 | N/A   |
| ResNet50 (Fine-Tuning)    | 97.33%        | N/A                 | N/A   |
| MobileNet (Feature Extraction) | 92.2%    | N/A                 | N/A   |
| MobileNet (Fine-Tuning)   | 97.00%        | N/A                 | N/A   |
| VGG16 (Feature Extraction) | 95.77%       | N/A                 | N/A   |
| VGG16 (Fine-Tuning)      | 99.94%        | N/A                 | N/A   |

