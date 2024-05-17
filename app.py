import streamlit as st
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import MeanSquaredError

# Custom objects for loading the model
custom_objects = {"MeanSquaredError": MeanSquaredError}

# Load the model with custom objects
model = load_model('pkl/Model.h5', custom_objects=custom_objects)

# Load the book data
with open('pkl/books.json', 'r') as f:
    books = json.load(f)

# Load the tokenizer
with open('pkl/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Extract book titles and create mappings
book_titles = [book[0] for book in books]
book_index = {book[0]: idx for idx, book in enumerate(books)}
index_book = {idx: book[0] for idx, book in enumerate(books)}

# Streamlit app
st.title('Book Recommendation System')
st.write('Enter a book title to get recommendations.')

# User input
book_input = st.text_input('Book Title')

if book_input:
    if book_input in book_index:
        # Find the index of the book
        book_idx = book_index[book_input]
        
        # Tokenize and pad the input book title
        seq = tokenizer.texts_to_sequences([book_input])
        padded_seq = pad_sequences(seq, maxlen=model.input_shape[1], padding='post')
        
        # Get the embeddings for the input book
        book_embedding = model.predict(padded_seq)
        
        # Find similar books using cosine similarity
        embeddings = model.predict(pad_sequences(tokenizer.texts_to_sequences(book_titles), maxlen=model.input_shape[1], padding='post'))
        similarities = np.dot(embeddings, book_embedding.T).flatten()
        similar_indices = similarities.argsort()[-6:][::-1][1:]
        
        # Display recommendations
        st.write('Recommendations:')
        for idx in similar_indices:
            st.write(f"{index_book[idx]}")
    else:
        st.write("Book not found. Please enter a valid book title.")
