import numpy as np
from math import log
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import time
from scipy import sparse
from scipy.sparse import csr_matrix
import os
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfTransformer

dimension = 100
# test_bag_of_words = load_svmlight_file("aclImdb/test/labeledBow.feat")
# train_bag_of_words = load_svmlight_file("aclImdb/train/labeledBow.feat")

# X_train = train_bag_of_words[0]
# X_test = test_bag_of_words[0]
# Y_train = train_bag_of_words[1]
# Y_test = test_bag_of_words[1]

# Y_train = (Y_train > 4).astype(int)
# Y_test = (Y_test > 4).astype(int)

# start_time = time.time()
# X_train = sparse.csc_matrix(X_train)
# X_train = X_train[:, : 89523]
# X_test = sparse.csc_matrix(X_test)


# tf_transformer = TfidfTransformer()
# X_train_tfidf = tf_transformer.fit_transform(X_train)
# X_test_tfidf = tf_transformer.fit_transform(X_test)



# if not os.path.exists("glove_arrays"):
#     os.makedirs("glove_arrays")

# # vocab = np.loadtxt("aclImdb/imdb.vocab")
# text_file = open("aclImdb/imdb.vocab", "r")
# vocab = text_file.read().splitlines()#readlines().strip()
# vocab = {k:v for v,k in enumerate(vocab)}
# print(type(vocab), vocab[:4])
# exit()

# def loadGloveModel(gloveFile):
#     print("Loading vectors")
#     f = open(gloveFile,'r')
#     model = {}
#     for line in f:
#         # split the lines to make a list
#         splitLine = line.split()
#         # store the first element as a word
#         word = splitLine[0]
#         # store the corresponding vector as embedding
#         embedding = np.array([float(val) for val in splitLine[1:]])
#         # store the key value pair of word and vector in a dictionary
#         model[word] = embedding
#     print("Done.",len(model)," words loaded!")
#     return model

# word_vec = loadGloveModel("glove.6B/glove.6B." + str(dimension) + "d.txt")

# X_train = []
# Y_train = []
# X_test = []
# Y_test = []

# print("\nCreating average vectors..")
# start_time = time.time()
# roots = ["aclImdb/train/neg", "aclImdb/train/pos", "aclImdb/test/neg", "aclImdb/test/pos"]
# Y_train_neg = np.full(len(os.listdir(root)), 1)
# for root in roots:
#     i = 0
#     for filename in os.listdir(root):
#         with open(root+"/"+filename) as f:
#             if i%500 ==0:
#                 print(i)
#             i += 1
#             raw = f.read()
#             tokens = word_tokenize(raw)
#             sum_vec = np.zeros(dimension)
#             count = 0
#             for token in tokens:
#                 try:
#                     vec = word_vec[token.lower()]
#                     # print("embbedding : ", type(vec))
#                     find_index = vocab[token.lower()]
#                     doc_id = int(filename.split('_')[0])
#                     if "train" in root:
#                         tfidf_val = X_train_tfidf[doc_id, find_index]
#                     else:
#                         tfidf_val = X_test_tfidf[doc_id, find_index]
#                     vec = vec*tfidf_val
#                     count += 1
#                     # print(vec)
#                 except Exception as e:
#                     # print(e)
#                     continue

#                 sum_vec = sum_vec + vec
#                 # print(sum_vec)
#             if count != 0:
#                 avg_vec = sum_vec / count
#             else :
#                 avg_vec = sum_vec
#             if "train" in root:
#                 X_train.append(avg_vec)
#                 Y_train.append(1) if "pos" in root else Y_train.append(0)
#             else:
#                 X_test.append(avg_vec)
#                 Y_test.append(1) if "pos" in root else Y_test.append(0)
#             # break

# print("Average vectors calculated in %d Seconds" % int(time.time()-start_time))


# X_train = np.array(X_train)
# Y_train = np.array(Y_train)
# X_test = np.array(X_test)
# Y_test = np.array(Y_test)
# np.save("glove_arrays/X_train_glove_"+ str(dimension)+ ".npy", X_train)
# np.save("glove_arrays/Y_train_glove_"+ str(dimension)+ ".npy", Y_train)
# np.save("glove_arrays/X_test_glove_"+ str(dimension)+ ".npy", X_test)
# np.save("glove_arrays/Y_test_glove_"+ str(dimension)+ ".npy", Y_test)

X_train = np.load("glove_arrays/X_train_glove_"+ str(dimension)+ ".npy")
Y_train = np.load("glove_arrays/Y_train_glove_"+ str(dimension)+ ".npy")
X_test = np.load("glove_arrays/X_test_glove_"+ str(dimension)+ ".npy")
Y_test = np.load("glove_arrays/Y_test_glove_"+ str(dimension)+ ".npy")

print("Training...")
# clf = BernoulliNB()
# clf = LogisticRegression()
clf = svm.SVC(verbose=True)
# clf = MLPClassifier(learning_rate_init=0.001, verbose=True, max_iter=200, hidden_layer_sizes=(100, ))
start_time = time.time()
clf.fit(X_train, Y_train)
print("Training completed in %d Seconds" % int(time.time()-start_time))

start_time = time.time()
Y_pred = clf.predict(X_test)
print("Testing completed in %d Seconds" % int(time.time()-start_time))
print(accuracy_score(Y_test, Y_pred))
