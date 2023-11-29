from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Dataset
dataset = {
    "What is the capital of France?": "Paris",
    "Who wrote the play Romeo and Juliet?": "William Shakespeare",
    "What is the largest planet in our solar system?": "Jupiter",
    "What programming language is often used for machine learning?": "Python",
    "What is the process of converting text data into numerical data called?": "Text encoding",
}

# Extract questions and answers
questions = list(dataset.keys())
answers = list(dataset.values())

# Vectorization
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Nearest Neighbor model
model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
model.fit(question_vectors)

# Function to get answer
def get_answer(query):
    query_vec = vectorizer.transform([query])
    _, indices = model.kneighbors(query_vec)
    closest_question_index = indices[0][0]
    return answers[closest_question_index]

# Example usage
query = "Romeo and Juliet"
print("Question:", query)
print("Answer:", get_answer(query))
