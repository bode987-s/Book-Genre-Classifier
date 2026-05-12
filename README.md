# Multimodal Book Classification

This project classifies books using: 

1. models: 

1.1 TEXT
    ├── Clean Text
    ├── TF-IDF 
    │       ├── Logistic Regression يوسف رجب  
    │       └── SVM  محمود عماد
    └── Tokenization → Embedding → LSTM محمود محمد 

1.2 IMAGE
    ├── Resize (224×224)
    ├── CNN يوسف عبدالحليم 
    └── ResNet50 مصطفي فؤاد

1.3 FUSION عبدالرحمن بدوي 
    ├── Text Features (TF-IDF)
    ├── Image Features (CNN + ResNet)
    └── Random Forest

1.4 EVALUATION 
    ├── Accuracy
    ├── Precision / Recall / F1
    └── Confusion Matrix

2. Scrapped Dataset
   Books dataset containing:
- title
- description
- cover image
- category

3. Technologies
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy

4. Kaggle workspace for model training: using GPU P100
