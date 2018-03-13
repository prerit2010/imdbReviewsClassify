import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import time
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

test_bag_of_words = load_svmlight_file("aclImdb/test/labeledBow.feat")
train_bag_of_words = load_svmlight_file("aclImdb/train/labeledBow.feat")

X_train = train_bag_of_words[0]
X_test = test_bag_of_words[0]
Y_train = train_bag_of_words[1]
Y_test = test_bag_of_words[1]

start_time = time.time()
counts_train = np.array([ i.count_nonzero() for i in X_train ])
counts_test = np.array([ i.count_nonzero() for i in X_test ])
print("Counts completed in %d Seconds" % int(time.time()-start_time))

start_time = time.time()
r,c = X_train.nonzero()
val = np.repeat(1.0/counts_train, X_train.getnnz(axis=1))
rD_sp = csr_matrix((val, (r,c)), shape=(X_train.shape))
X_train = X_train.multiply(rD_sp)

r,c = X_test.nonzero()
val = np.repeat(1.0/counts_test, X_test.getnnz(axis=1))
rD_sp = csr_matrix((val, (r,c)), shape=(X_test.shape))
X_test = X_test.multiply(rD_sp)

# r,c = X_train.nonzero()
# rD_sp = csr_matrix(((1.0/counts_train)[r], (r,c)), shape=(X_train.shape))
# X_train = X_train.multiply(rD_sp)

# r,c = X_test.nonzero()
# rD_sp = csr_matrix(((1.0/counts_test)[r], (r,c)), shape=(X_test.shape))
# X_test = X_test.multiply(rD_sp)


# X_train = X_train / counts_train[:,None]
# X_test = X_test / counts_test[:,None]
print("Divisions completed in %d Seconds" % int(time.time()-start_time))

start_time = time.time()
Y_train = np.array([0 if i <= 4 else 1 for i in Y_train])
Y_test = np.array([0 if i <= 4 else 1 for i in Y_test])
print("Binary conversion completed in %d Seconds" % int(time.time()-start_time))

start_time = time.time()
X_train = sparse.csc_matrix(X_train)
X_train = X_train[:, : 89523]
X_test = sparse.csc_matrix(X_test)
print("Transformation completed in %d Seconds" % int(time.time()-start_time))
# print(type(X_train))
# print(X_train.shape)
print("Training...")
# clf = BernoulliNB()
clf = LogisticRegression()
# clf = svm.SVC(max_iter=1000)
# clf = MLPClassifier(learning_rate_init=0.001, verbose=True, max_iter=50, hidden_layer_sizes=(20, ))
start_time = time.time()
clf.fit(X_train, Y_train)
print("Training completed in %d Seconds" % int(time.time()-start_time))

start_time = time.time()
Y_pred = clf.predict(X_test)
print("Testing completed in %d Seconds" % int(time.time()-start_time))
print(accuracy_score(Y_test, Y_pred))
