import pandas as pd
import re
import nltk
import streamlit as st
import matplotlib.pyplot as plt
import os

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "fba_inventory.csv")
data = pd.read_csv(file_path)

text_column = "product_name"
label_column = "recommended_action"
data = data.dropna(subset=[label_column])

def clean_text(text):
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я\s]", "", text)
    text = text.lower()
    return text

data[text_column] = data[text_column].astype(str).apply(clean_text)
nltk.download("punkt")
data["tokens"] = data[text_column].apply(nltk.word_tokenize)

nltk.download("stopwords")
stop_words = set(stopwords.words("russian"))
data["tokens"] = data["tokens"].apply(
    lambda x: [word for word in x if word not in stop_words]
)

X_train, X_test, y_train, y_test = train_test_split(
    data[text_column], data[label_column], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vect, y_train)

y_pred = model.predict(X_test_vect)

st.title("Анализ текста")
input_text = st.text_area("Введите текст для анализа:")

if st.button("Анализировать"):
    cleaned_text = clean_text(input_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    st.write(f"Предсказание: {prediction}")

    st.subheader("Распределение предсказанных меток")
    pred_counts = pd.Series(y_pred).value_counts()
    fig2, ax2 = plt.subplots()
    ax2.bar(pred_counts.index, pred_counts.values)
    ax2.set_xticks(list(pred_counts.index))
    ax2.set_xticklabels(pred_counts.index, rotation=90)
    st.pyplot(fig2)

st.subheader("Распределение меток в обучающих данных")
label_counts = data[label_column].value_counts()
fig, ax = plt.subplots()

ax.bar(label_counts.index, label_counts.values)
ax.set_xticks(list(label_counts.index))
ax.set_xticklabels(label_counts.index, rotation=90)

st.pyplot(fig)
st.subheader("Таблица данных")
st.dataframe(data[[text_column, label_column]])