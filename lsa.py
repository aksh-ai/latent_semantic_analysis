import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

def tokenizer(sentence):
    sentence = sentence.lower()
    tokens = nltk.tokenize.word_tokenize(sentence)
    tokens = [token for token in tokens if len(token)>2]
    tokens = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [token for token in tokens if not any(c.isdigit() for c in token)]
    return tokens

def vectorizer(tokens):
    x = np.zeros(len(index_map))
    for token in tokens:
        index = index_map[token]
        x[index] = 1
    return x    

wordnet_lemmatizer = WordNetLemmatizer()

titles = [line.rstrip() for line in open('dataset/books.txt')]

stopwords = set(word.rstrip() for word in open('stopwords.txt'))

index_map = {}
current_idx = 0
all_tokens = []
all_titles = []
word_map = []

for title in titles:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(title)
        tokens = tokenizer(title)
        all_tokens.append(tokens)

        for token in tokens:
            if token not in index_map:
                index_map[token] = current_idx
                current_idx += 1
                word_map.append(token)

    except Exception as e:
        print("Error")
        #print(e)
        #break
        pass

print("Tokenized...")

N = len(all_tokens)
D = len(index_map)
X = np.zeros((D, N))
i = 0 

for tokens in all_tokens:
    X[:, i] = vectorizer(tokens)
    i += 1

print("Vectorized...")

svd = TruncatedSVD()

Z = svd.fit_transform(X)

print("Decomposed successfully...\nPlotting...\n")

plt.scatter(Z[:, 0], Z[:, 1])
for i in range(D):
    plt.annotate(s=word_map[i], xy=(Z[i, 0], Z[i, 1]))

plt.show()