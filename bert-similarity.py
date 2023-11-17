import dummy_data

import sentence_transformers
from sentence_transformers import util

model = sentence_transformers.SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Example paragraphs
paragraph1 = constants.para1
paragraph2 = constants.para3


# Lowercasing
paragraph1 = paragraph1.lower()
paragraph2 = paragraph2.lower()
# print(paragraph1)

# Get paragraph embeddings
embeddings = model.encode([paragraph1, paragraph2])

# Calculate cosine similarity
similarity = sentence_transformers.util.pytorch_cos_sim(embeddings[0], embeddings[1])
print(similarity.item())

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Convert to lowercase
    tokens = [token.lower() for token in tokens]

    tagged_tokens = pos_tag(tokens)
    tokens = [word for word, pos in tagged_tokens if
              not (pos.startswith('VB') or pos.startswith('CONJ') or pos.startswith('ADP') or pos.startswith('ADV'))]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    print("\n length", len(tokens))

    # Lemmatization using WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove non-alphabetic characters
    # tokens = [token for token in tokens if token.isalpha()]

    preprocessed_text = ' '.join(tokens)

    sentences = sent_tokenize(preprocessed_text)
    print("\nsentences", sentences)
    print("\npreprocessed_text", preprocessed_text)
    return preprocessed_text
