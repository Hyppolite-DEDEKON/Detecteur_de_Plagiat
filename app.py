import nltk
from flask import Flask, render_template, request
from utils import preprocess_text, calculate_similarity

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_plagiarism', methods=['POST'])
def detect_plagiarism():
    text1 = request.form['text1']
    text2 = request.form['text2']

    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)

    similarity_percentage = calculate_similarity(preprocessed_text1, preprocessed_text2)

    return f"Ressemblance: {similarity_percentage:.2f}%"

if __name__ == '__main__':
    app.run(debug=True)
