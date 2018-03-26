import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import time
from scipy import sign
from scipy.sparse import csr_matrix

test_bag_of_words = load_svmlight_file("aclImdb/test/labeledBow.feat")
train_bag_of_words = load_svmlight_file("aclImdb/train/labeledBow.feat")

X_train = train_bag_of_words[0]
Y_train = train_bag_of_words[1]
X_test = test_bag_of_words[0]
Y_test = test_bag_of_words[1]
X_train = (X_train != 0).astype(int) # Convert bag of words to binary bag of words.
# print(X_train[0])
print(X_train[0].count_nonzero())
Y_train = np.array([0 if i <= 4 else 1 for i in Y_train])
Y_test = np.array([0 if i <= 4 else 1 for i in Y_test])
X_train = X_train[:, : 89523]

print("Training...")
# clf = BernoulliNB()
# clf = LogisticRegression()
clf = svm.SVC(verbose=True)
# clf = MLPClassifier(learning_rate_init=0.001, verbose=True, max_iter=50, hidden_layer_sizes=(10, ))
start_time = time.time()
clf.fit(X_train, Y_train)
print("Training completed in %d Seconds" % int(time.time()-start_time))

start_time = time.time()
Y_pred = clf.predict(X_test)
print("Testing completed in %d Seconds" % int(time.time()-start_time))
print(accuracy_score(Y_test, Y_pred))
