# Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 1.Load and Preprocess Data: Load the dataset, select relevant columns, and map labels to numeric values.

2.Split Data: Divide the dataset into training and test sets.

3.Vectorize Text: Convert text messages into a numerical format using TfidfVectorizer.

4.Train SVM Model: Fit an SVM model with a linear kernel on the training data.

5.Evaluate Model: Predict on the test set and print the accuracy score and classification report.

6.Visualize Results: Plot the confusion matrix, ROC curve, and precision-recall curve for detailed evaluation.

## Program:


```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Zafreen J
RegisterNumber:  212223040252
*/
```



```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  # Select relevant columns
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Map labels to 0 (ham) and 1 (spam)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)
y_proba = model.decision_function(X_test_vec)  # For ROC and Precision-Recall curves

# 1. Accuracy Score
print("Accuracy:", accuracy_score(y_test, y_pred))

# 2. Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 4. ROC Curve and AUC Score
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# 5. Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


```


## Output:


![Screenshot 2024-11-08 200645](https://github.com/user-attachments/assets/03b77418-ded4-4a9b-b7a7-c5d18e71d21b)


![Screenshot 2024-11-08 200724](https://github.com/user-attachments/assets/e2e9061e-3fe8-4245-9047-21dece8232f3)


![Screenshot 2024-11-08 200748](https://github.com/user-attachments/assets/0cfae893-8e39-4da4-b105-bed39e8b41e4)


![Screenshot 2024-11-08 200811](https://github.com/user-attachments/assets/0c74b76f-ccb8-4282-9dee-3459474574ee)


![Screenshot 2024-11-08 200829](https://github.com/user-attachments/assets/dae83d3b-a8c2-43ea-b2c3-50fdc6b26833)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
