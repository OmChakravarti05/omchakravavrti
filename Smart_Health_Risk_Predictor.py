import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("E:\insurance.csv")
df.head()

df.info()
df.describe()
df.isnull().sum()

sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Insurance Charges vs Smoker')
plt.show()

sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df)
plt.title('BMI vs Charges')
plt.show()

# Feature engineering
df['high_risk'] = (df['charges'] > 16000).astype(int)
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Model training
X = df.drop(columns=['charges', 'high_risk'])
y = df['high_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))

# Feature importance
feat_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

sns.barplot(x='Importance', y='Feature', data=feat_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

