

import streamlit as st
from keras.models import load_model
import pickle
import numpy as np
import heapq
from nltk.tokenize import RegexpTokenizer

# Loading the model and history
model = load_model('C:\\Users\\User\\Downloads\\next_word_model.h5')
history = pickle.load(open("C:\\Users\\User\Downloads\\history (1).p", "rb"))

# Loading unique words and corresponding index
unique_words = np.load('C:\\Users\\User\\Downloads\\unique_words.npy')
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

# Defining the functions for input preparation and prediction
def prepare_input(text, selected_word=None):
    if selected_word:
        text = ' '.join(text.split()[1:] + [selected_word])
    x = np.zeros((1, 5, len(unique_words)))
    for t, word in enumerate(text.split()):
        x[0, t, unique_word_index[word]] = 1
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completions(text):
    if text == "":
        return []
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    return list(zip(unique_words, preds))

# Setting up the Streamlit app
st.set_page_config(page_title="Next Word Prediction", page_icon="🔮", layout="wide")

st.title("Next Word Prediction")

# Taking input from user
input_text = st.text_input("Enter a sentence to predict the next word:", "")

if input_text:
    # Tokenizing the input and selecting the last 5 words
    tokenizer = RegexpTokenizer(r'\w+')
    input_words = tokenizer.tokenize(input_text.lower())
    seq = " ".join(input_words[-5:])

    # Getting the predictions and showing the top 5 options
    predictions = predict_completions(seq)
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    st.write("Next possible words:")
    for i, pred in enumerate(sorted_preds[:5]):
        st.write("{}) {} ({:.2f}%)".format(i+1, pred[0], pred[1]*100))

    # Letting the user select the predicted word(s) and updating the input
    selected_indices = st.text_input("Enter the numbers of the selected words (separated by comma, or 0 to exit):", "")
    if selected_indices != "" and selected_indices != "0":
        try:
            selected_indices = [int(x)-1 for x in selected_indices.split(",")]
            selected_words = [sorted_preds[i][0] for i in selected_indices]
            input_text += " "+' '.join(selected_words)
            st.write("Selected words:", selected_words)
            st.write("Full sentence:", input_text)
        except:
            st.write("Invalid input. Try again.")