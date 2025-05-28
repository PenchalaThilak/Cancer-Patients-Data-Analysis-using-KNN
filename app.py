import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
dataset = pd.read_csv(r"C:\\Users\\thila\\Downloads\ PYTHON FSDS NOTES\\JANUARY\\4TH JAN\\4th\\4th\\projects\\KNN\\brest cancer.txt")

# Renaming the Column Names
column_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 
                'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 
                'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
dataset.columns = column_names

# Drop ID column
dataset.drop('Id', axis=1, inplace=True)

# Convert 'Bare_Nuclei' to numeric
dataset['Bare_Nuclei'] = pd.to_numeric(dataset['Bare_Nuclei'], errors='coerce')

# Drop rows with missing values
dataset.dropna(inplace=True)

# Check class balance
sns.countplot(data=dataset, x='Class')
plt.title("Cancer Class Distribution")
plt.show()

# Feature and label separation
X = dataset.drop('Class', axis=1)
y = dataset['Class']

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
