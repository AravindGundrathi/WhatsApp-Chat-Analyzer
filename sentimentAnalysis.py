import re
import pickle
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util

nltk.download('movie_reviews')
nltk.download('punkt')

def clean(words):
    return dict([(word, True) for word in words])

# Load movie reviews data
negative_ids = movie_reviews.fileids('neg')
positive_ids = movie_reviews.fileids('pos')

negative_features = [(clean(movie_reviews.words(fileids=[f])), 'negative') for f in negative_ids]
positive_features = [(clean(movie_reviews.words(fileids=[f])), 'positive') for f in positive_ids]

# Split data into training and test sets
negative_cutoff = int(len(negative_features) * 0.95)
positive_cutoff = int(len(positive_features) * 0.90)

train_features = negative_features[:negative_cutoff] + positive_features[:positive_cutoff]
test_features = negative_features[negative_cutoff:] + positive_features[positive_cutoff:]

# Train a NaiveBayes classifier
classifier = NaiveBayesClassifier.train(train_features)
print('Training complete')
print('accuracy:', nltk.classify.util.accuracy(classifier, test_features) * 100, '%')
classifier.show_most_informative_features()

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

def load_classifier(model_path='model.pkl'):
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    return classifier

def perform_sentiment_analysis(chat_content):
    classifier = load_classifier('model.pkl')

    output = ""
    pos_count = 0
    neg_count = 0

    pattern = re.compile(r'(\d{2}/\d{2}/\d{2}), (\d{2}:\d{2}) - ([^:]+): (.+)')

    for match in pattern.finditer(chat_content):
        date, time, name, chat = match.groups()
        res = classifier.classify(clean(word_tokenize(chat)))
        sentiment_color = 'green' if res == 'positive' else 'red'
        bsentiment_color = '#e6ffe8' if res == 'positive' else '#ffe8e6'
        output += f"<p><span style='color:brown; font-size:20px; font-weight:bold; display:inline-block; width:80px;'>{name} :</span> <span style='border: 1px solid {sentiment_color}; padding: 8px; margin: 2px; border-radius: 5px; background-color: {bsentiment_color};display:inline-block; width:600px;'>{chat}</span></p>\n"
        if res == 'positive':
            pos_count += 1
        else:
            neg_count += 1

    return output.strip(), pos_count, neg_count

# Read the chat content from the file
with open('use.txt', 'r', encoding='utf-8') as file:
    chat_content = file.read()

output, pos_count, neg_count = perform_sentiment_analysis(chat_content)
if output:
    print(output)
    print("Total Positives:", pos_count)
    print("Total Negatives:", neg_count)
else:
    print("Error occurred while processing the sentiment analysis.")







