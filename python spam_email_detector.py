# spam_email_detector.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------
# STEP 1: Sample Dataset
# -----------------------

data = {
    'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
    'text': [
        "Hey, are we still meeting tomorrow?",
        "Congratulations! You've won a free car. Click here to claim.",
        "Don't forget to bring your notebook.",
        "URGENT! Your mobile number has won $1000!",
        "I'll call you in 5 minutes.",
        "Get rich quick! Work from home, earn ₹5000/day.",
        "Meeting is rescheduled to 3 PM.",
        "Claim your reward now. Limited time offer!"
    ]
}

df = pd.DataFrame(data)

# -----------------------
# STEP 2: Preprocessing
# -----------------------

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# -----------------------
# STEP 3: TF-IDF Vectorizer
# -----------------------

vectorizer = TfidfVectorizer()
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

# -----------------------
# STEP 4: Train Model
# -----------------------

model = MultinomialNB()
model.fit(X_train_tf, y_train)

# -----------------------
# STEP 5: Evaluate Model
# -----------------------

y_pred = model.predict(X_test_tf)
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------
# STEP 6: Predict Samples
# -----------------------

sample_msgs = [
    "You've been selected for a free gift card!",
    "Hi mom, just checking in.",
    "Urgent: Account will be suspended if not verified",
    "Let's meet at the library at 4 PM"
]

sample_tf = vectorizer.transform(sample_msgs)
predictions = model.predict(sample_tf)

print("\n📨 Sample Predictions:")
for msg, label in zip(sample_msgs, predictions):
    print(f"{'SPAM' if label else 'HAM'} ➜ {msg}")
