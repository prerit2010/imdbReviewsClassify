import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import time, os
from keras.utils import to_categorical

dimension = 100
def loadGloveModel(gloveFile):
    print("Loading vectors")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        # split the lines to make a list
        splitLine = line.split()
        # store the first element as a word
        word = splitLine[0]
        # store the corresponding vector as embedding
        embedding = np.array([float(val) for val in splitLine[1:]])
        # store the key value pair of word and vector in a dictionary
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

word_vec = loadGloveModel("glove.6B/glove.6B." + str(dimension) + "d.txt")

Y_train = np.concatenate((np.full(12500, 0), np.full(12500, 1)))
Y_test = Y_train


print("\nCreating average vectors..")
start_time = time.time()
roots = ["aclImdb/train/neg", "aclImdb/train/pos"]

# Y_train_neg = np.full(len(os.listdir(root)), 1)

texts_train = []
labels_index_train = {}
labels_train = []

for root in roots:
    i = 0
    for filename in os.listdir(root):
        with open(root+"/"+filename) as f:
            if i%5000 ==0:
                print(i)
            i += 1
            label_id = len(labels_index_train)
            labels_index_train[filename] = label_id
            raw = f.read()
            texts_train.append(raw)
            labels_train.append(label_id)

roots = ["aclImdb/test/neg", "aclImdb/test/pos"]

texts_test = []
labels_index_test = {}
labels_test = []

for root in roots:
    i = 0
    for filename in os.listdir(root):
        with open(root+"/"+filename) as f:
            if i%5000 ==0:
                print(i)
            i += 1
            label_id = len(labels_index_test)
            labels_index_test[filename] = label_id
            raw = f.read()
            texts_test.append(raw)
            labels_test.append(label_id)

print("Average vectors calculated in %d Seconds" % int(time.time()-start_time))

MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS) #nb_words=MAX_NB_WORDS
tokenizer.fit_on_texts(texts_train)
sequences = tokenizer.texts_to_sequences(texts_train)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

Y_train = to_categorical(Y_train, num_classes=2)
print('Shape of data tensor:', X_train.shape)
print('Shape of label tensor:', Y_train.shape)
embedding_dimension = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
for word, i in word_index.items():
    embedding_vector = word_vec.get(word.lower())
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            embedding_dimension,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)

lstm_out = 3
#Setting hidden layer size

model = Sequential()
model.add(embedding_layer)
#Add Embedding Layer

model.add(LSTM(lstm_out,dropout_U=0.2,dropout_W=0.2))
model.add(Dense(2,activation='softmax'))
#Add hidden layer

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#Create LSTM model
print(model.summary())

batch_size = 50
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 1)
#Train the model


#Predict accuracy

#Test
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS) #nb_words=MAX_NB_WORDS
tokenizer.fit_on_texts(texts_test)
sequences = tokenizer.texts_to_sequences(texts_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
Y_test = to_categorical(Y_test, num_classes=2)
# labels_test = to_categorical(np.asarray(labels_test))
print('Shape of data tensor:', X_test.shape)
print('Shape of label tensor:', Y_test.shape)


score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)
print(acc)

# split the data into a training set and a validation set
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]
# nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
