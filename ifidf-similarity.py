from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string
import re

import dummy_data


nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Remove date-like strings using regular expression
    # text = re.sub(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b', '', text)

    text=text.lower()
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove punctuation
    words = [word for word in words if word.isalnum()]

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Join the processed words back into a sentence
    processed_text = ' '.join(words)

    return processed_text

def calculate_tfidf(texts):
    # Preprocess each text
    preprocessed_texts = [preprocess_text(text) for text in texts]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
    return tfidf_matrix, vectorizer

def calculate_cosine_similarity(tfidf_matrix_job, tfidf_matrix_resumes):
    similarity_matrix = cosine_similarity(tfidf_matrix_job, tfidf_matrix_resumes)
    return similarity_matrix

def main():
    # Job Description
    job_description =dummy_data.jd_cloud_dev
    # Resumes
    resumes = [
        dummy_data.resume_rahul,
        dummy_data.resume_full_stack_ashwin
    ]

    # Calculate TF-IDF matrix for job description and get the vectorizer
    tfidf_matrix_job, vectorizer = calculate_tfidf([job_description])

    # Use the same vectorizer to transform resumes
    tfidf_matrix_resumes = vectorizer.transform(resumes)

    # Calculate cosine similarity matrix
    similarity_matrix = calculate_cosine_similarity(tfidf_matrix_job, tfidf_matrix_resumes)

    # Print the similarity matrix
    print("Cosine Similarity Matrix:")
    print(similarity_matrix)

    # Print individual similarity scores for each resume
    for i, score in enumerate(similarity_matrix[0]):
        print(f"Similarity with Resume {i + 1}: {score}")

if __name__ == "__main__":
    main()
