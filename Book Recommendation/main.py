import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import keybert
from keybert import KeyBERT
import pandas as pd
from flask import Flask, request, render_template
import json

book_input = ""
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


# # Read the data
# data = pd.DataFrame({
#     'title': ['Book1', 'Book2', 'Book3', 'Book4'],
#     'description': [
#         'It is book based on machine learning technology.',
#         'A heartwarming tale of friendship and courage set in a small town in rural America.',
#         'A heartwarming tale of friendship and courage set in a small town in rural America.',
#         'A heartwarming tale of friendship and courage set in a small town in rural America.'
#     ],
#     'ratings': ['3', '2', '4', '5']
# })
# # Check if any book with the entered book name or keyword is available
# book_indices = pd.Series(data.index, index=data['title'])
keyword_strings = []
# Get the data in the desired format
data = pd.read_csv('Sample_Data1.csv', usecols=['title', 'description', 'ratings'])
data = pd.DataFrame({
    'title': data['title'],
    'description': data['description'],
    'ratings': data['ratings']
})
book_indices = pd.Series(data.index, index=data['title'])


# Filter the description by getting all tokens and the eliminating the stop words
def token():
    # Tokenize the descriptions
    tokens = []
    for i, row in data.iterrows():
        desc = row['description']
        type(desc)
        desc_tokens = word_tokenize(desc)
        tokens.append(desc_tokens)

    # Remove stop words and lemmatize the tokens
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    filtered_tokens = []
    for desc_tokens in tokens:
        filtered_tokens.append(
            [lemmatizer.lemmatize(word.lower()) for word in desc_tokens if word.lower() not in stop_words])
    return filtered_tokens


# Keyword extraction
def keyword():
    # Extract keywords
    model = KeyBERT('distilbert-base-nli-mean-tokens')
    keywords = []
    filtered_tokens = token()
    for i, tokens in enumerate(filtered_tokens):
        doc = ' '.join(tokens)
        doc_keywords = model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words='english', use_maxsum=True,
                                              nr_candidates=20, top_n=5)
        keyword_list = [key for key, _ in doc_keywords]
        synonyms = []
        for keyword in keyword_list:
            synsets = wordnet.synsets(keyword)
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonyms.append(lemma.name())
        synonyms = list(set(synonyms))
        keyword_list.extend(synonyms)
        keywords.append(keyword_list)
    return keywords


# Check if book is available as book name or similar to the entered keyword
def check_book_availability(book_input):
    keywords = keyword()
    if book_input in data['title'].values:
        book_available = True
        book_title = book_input
        # hr = recommend_book()
        print(book_input, 'is available')
        return vectorization(book_available)
    elif any(book_input.lower() in ' '.join(keywords[i]).lower() for i in range(len(keywords))):
        related_book_available = True
        print('Books related to [', book_input, '] are available')
        return vectorization(related_book_available)
    else:
        print('Book not available')
        return False


# Vectorization and cosine similarities
def vectorization(flag):
    if flag:
        keywords = keyword()
        for keywords_list in keywords:
            keyword_string = ' '.join(keywords_list)
            keyword_strings.append(keyword_string)
        # Vectorize descriptions and calculate cosine similarity
        tfidf = TfidfVectorizer()
        desc_tfidf = tfidf.fit_transform(keyword_strings)
        cosine_sim = cosine_similarity(desc_tfidf)

        # Create dictionary with book titles as keys and keywords, ratings as values
        keyword_data = {}
        for i in range(len(data['title'])):
            keyword_data[data['title'][i]] = keywords[i], [data['ratings'][i]]
        print("keyword_data:", keyword_data)
        return recommend_book(cosine_sim, keyword_data)


# Recommend top 3 Similar books and sort by highest ratings
def recommend_book(cosine_sim, keyword_data):
    # Get the index of the input book or keyword
    if book_input not in book_indices:
        keys = [key for key in keyword_data.keys() if book_input.lower() in keyword_data[key][0]]
        book_detail = {}
        for i in keys:
            for key, value in keyword_data.items():
                if i == key:
                    ratings = [int(r) for r in value[1]]
                    book_detail[i] = max(ratings)
        sorted_books = list(sorted(book_detail.items(), key=lambda x: x[1], reverse=True))
        # print(f"Please find the recommended books for {book_input}:", sorted_books)
        sorted_books_list = [{'title': book[0], 'ratings': str(book[1])} for book in sorted_books]
        return sorted_books_list

    else:
        book_idx = book_indices[book_input]  # gets the ID of book to find cosine similarity
        print("book_idx", book_idx)

        # Get the cosine similarity scores for the input book
        sim_scores = list(enumerate(cosine_sim[book_idx]))
        # print(sim_scores)

        # Sort the book indices based on the cosine similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        top_books_indices = [i[0] for i in sim_scores if i[1] > 0.7][:3]

        print("top_books_indices", top_books_indices)
        top_books = data.loc[top_books_indices, ['title', 'ratings']]
        # Sort the books by ratings in descending order
        top_books = top_books.sort_values('ratings', ascending=False)
        # Get the recommended book titles (excluding the input book)
        recommended_books = top_books.loc[top_books['title'] != book_input, ['title', 'ratings']].to_dict('records')
        # print(type(recommended_books))
        if len(recommended_books) != 0:
            print("Books similar to", book_input, "are:", recommended_books)
        print("recommended_books", recommended_books)
        return recommended_books


@app.route('/recommend')
def recommended_books():
    global book_input
    book_input = request.args.get('book_input')
    books = check_book_availability(book_input)
    return render_template('result.html', book_input=book_input, books=books)


if __name__ == '__main__':
    app.run()
# Take book name as user's input
if len(book_input) != 0:
    check_book_availability()
else:
    print("Please enter the book name")
