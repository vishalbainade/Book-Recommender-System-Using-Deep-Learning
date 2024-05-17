Book Recommender System Using Deep Learning

Overview
This project is a Book Recommender System built using deep learning techniques. The system leverages a neural network to provide personalized book recommendations to users based on their historical reading data and preferences.

Features
Personalized Recommendations: Generates a list of books tailored to individual users.
User and Item Embeddings: Uses embeddings to represent users and books in a continuous vector space.
Neural Network Model: Employs a deep learning model to predict user ratings for books.
Scalable Architecture: Designed to handle a large dataset of users and books efficiently.

Requrements
Python 3.8
TensorFlow 2.0
Pandas
NumPy
Scikit-learn

Dataset
The recommender system uses a dataset that contains user-book interactions, including user ratings for books. Ensure your dataset is in the correct format:
found_books_filtered.ndjson

Model Architecture
The recommender system uses a neural collaborative filtering approach with the following architecture:

Embedding Layers: Separate embedding layers for users and books.
Dense Layers: Multiple dense layers to capture complex interactions between users and books.
Output Layer: A single neuron with a sigmoid activation function to predict the rating.

Hyperparameters
Key hyperparameters used in the model:

Embedding Dimension: Size of the embedding vectors for users and books. 
Dense Layers: Number and size of dense layers.
Activation Function: Activation functions used in dense layers.
Batch Size: Number of samples per batch.
Epochs: Number of training epochs.

Evaluation
The model is evaluated using metrics such as Mean Squared Error (MSE) and Root binary cross entropy to measure the accuracy of predicted ratings.

Contact
For any questions or suggestions, please contact vishalbainade390@gmail.com or open an issue on the GitHub repository.