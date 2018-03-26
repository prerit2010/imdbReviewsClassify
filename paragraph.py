import gensim
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
import uuid
import pickle

if not os.path.exists("paragraph_arrays"):
    os.makedirs("paragraph_arrays")

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,[self.labels_list[idx]])



print("\nCreating average vectors..")
start_time = time.time()
# roots = ["aclImdb/train/neg", "aclImdb/train/pos"] # , "aclImdb/test/neg", "aclImdb/test/pos"]
roots = ["aclImdb/test/neg", "aclImdb/test/pos"]
# Y_train_neg = np.full(len(os.listdir(root)), 1)
doc_labels = []
train_data = []
# train_data = []
Y_train = np.concatenate((np.full(12500, 0), np.full(12500, 1)))
Y_test = Y_train

# print(len(os.listdir("aclImdb/train/neg")), len(os.listdir("aclImdb/train/pos")) \
        # , len(os.listdir("aclImdb/test/neg")), len(os.listdir("aclImdb/test/pos")))

# for root in roots:
#     i = 0
#     for filename in os.listdir(root):
#         doc_labels.append(uuid.uuid4().hex[:6].upper())
#         with open(root+"/"+filename) as f:
#             if i%2000 ==0:
#                 print(i)
#             i += 1
#             raw = f.read()
#             tokens = word_tokenize(raw)
#             train_data.append(tokens)
#           #   if "train" in root:
#           #     X_train.append(avg_vec)
#           #     Y_train.append(1) if "pos" in root else Y_train.append(0)
#           # else:
#           #     X_test.append(avg_vec)
#           #     Y_test.append(1) if "pos" in root else Y_test.append(0)

# print("Average vectors calculated in %d Seconds" % int(time.time()-start_time))
            # if "train" in root:
            #   train_data.append(tokens)
            # else:
            #   test_data.append(tokens)
# print(len(train_data))
# # exit()
# with open('paragraph_arrays/test_data', 'wb') as fp:
#     pickle.dump(train_data, fp)

# with open('paragraph_arrays/test_doc_labels', 'wb') as fp:
#     pickle.dump(doc_labels, fp)

# fp = open ('paragraph_arrays/test_data', 'rb')
# train_data = pickle.load(fp)
# fp = open ('paragraph_arrays/test_doc_labels', 'rb')
# doc_labels = pickle.load(fp)

# print(len(doc_labels))


# it = LabeledLineSentence(train_data, doc_labels)

# model = gensim.models.Doc2Vec(vector_size=100, min_count=0, alpha=0.025, min_alpha=0.025)
# model.build_vocab(it)

# print("\ntraining..")
# model.train(it, total_examples=len(train_data), epochs=1)
# for epoch in range(10):
#     print('iteration '+str(epoch+1))
#     model.train(it, total_examples=len(train_data), epochs=model.iter)
# # saving the created model
    
#   # model.train(it) 
#     model.alpha -= 0.002
#     model.min_alpha = model.alpha
    
# model.save('doc2vec_test.model')
# print("model saved")

##################################################################################################

#loading the model
d2v_model_train = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
d2v_model_test = gensim.models.doc2vec.Doc2Vec.load('doc2vec_test.model')

docvecs_train = []
docvecs_test = []

fp = open ('paragraph_arrays/train_data', 'rb')
train_data = pickle.load(fp)
fp = open ('paragraph_arrays/train_doc_labels', 'rb')
doc_labels_train = pickle.load(fp)
fp = open ('paragraph_arrays/test_data', 'rb')
test_data = pickle.load(fp)
fp = open ('paragraph_arrays/test_doc_labels', 'rb')
doc_labels_test = pickle.load(fp)


for label in doc_labels_train:
    docvecs_train.append(d2v_model_train.docvecs[label])

for label in doc_labels_test:
    docvecs_test.append(d2v_model_test.docvecs[label])

# print(docvec)
print(len(docvecs_train), len(docvecs_test))
X_train = np.array(docvecs_train)
X_test = np.array(docvecs_test)


print("Training...")
# clf = BernoulliNB()
# clf = LogisticRegression()
clf = svm.SVC(verbose=True)
# clf = MLPClassifier(learning_rate_init=0.001, verbose=True, max_iter=200, hidden_layer_sizes=(500, ))
start_time = time.time()
clf.fit(X_train, Y_train)
print("Training completed in %d Seconds" % int(time.time()-start_time))

start_time = time.time()
Y_pred = clf.predict(X_test)
print("Testing completed in %d Seconds" % int(time.time()-start_time))
print(accuracy_score(Y_test, Y_pred))


