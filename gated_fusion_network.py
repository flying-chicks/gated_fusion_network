from keras.layers import (Input, Embedding, Dense, Conv1D, MaxPool1D, Flatten,
                          Dropout, concatenate, multiply, RepeatVector, Lambda)
from keras.regularizers import l1_l2

from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.externals import joblib
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from pandas import concat
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from numpy import argmax

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


def read(fname):
    with open(fname, 'rb') as fr:
        return joblib.load(fr)


vocab_size = 60230
embedding_size = 300
max_len = 400

n_filter = 250
kernel_size = 3
hidden_size = 250

batch_size = 32
epochs = 10

images_size = 4
leaves_size = 300

is_gate = True
is_fusion = True

n_pos_samples = 22000

embedding_matrix = read("/home/ygy/embedding_matrix.pkl")

texts = Input(shape=(max_len,), dtype='int32', name='texts')

embedded_texts = Embedding(output_dim=embedding_size, input_dim=vocab_size,
                           input_length=max_len, weights=[embedding_matrix],
                           trainable=True)(texts)

embedded_texts = Dropout(0.5)(embedded_texts)

conv_out = Conv1D(n_filter, kernel_size, padding='valid', activation='relu', strides=1)(embedded_texts)

pooled_out = MaxPool1D()(conv_out)

flatten_out = Flatten()(pooled_out)
texts_hidden = Dense(hidden_size, activation="relu")(flatten_out)
texts_hidden = Dropout(0.5)(texts_hidden)
texts_hidden = Dense(hidden_size, activation="relu")(texts_hidden)
texts_hidden = Dropout(0.5)(texts_hidden)
texts_out = Dense(100, activation='tanh', name='texts_out')(texts_hidden)

images = Input(shape=(images_size,), name='images')
images_out = Dense(4, activation='tanh', name='images_out')(images)

leaves = Input(shape=(leaves_size,), name="leaves")
leaves_hidden = Dense(250, activation="relu")(leaves)
leaves_hidden = Dropout(0.5)(leaves_hidden)
leaves_hidden = Dense(250, activation="relu")(leaves_hidden)
leaves_hidden = Dropout(0.5)(leaves_hidden)
leaves_out = Dense(100, activation='tanh', name='leaves_out')(leaves_hidden)

if is_gate:
    texts_gate = Dense(100, activation="sigmoid", name='texts_gate')(concatenate([images_out, leaves_out], axis=-1))
    images_gate = Dense(4, activation="sigmoid", name='images_gate')(concatenate([texts_out, leaves_out], axis=-1))
    leaves_gate = Dense(100, activation="sigmoid", name='leaves_gate')(concatenate([texts_out, images_out], axis=-1))

    texts_out = multiply([texts_out, texts_gate])
    images_out = multiply([images_out, images_gate])
    leaves_out = multiply([leaves_out, leaves_gate])


def kronecker_product(mat1, mat2):
    n1 = mat1.get_shape()[1]
    n2 = mat2.get_shape()[1]
    mat1 = RepeatVector(n2)(mat1)
    mat1 = concatenate([mat1[:, :, i] for i in range(n1)], axis=-1)
    mat2 = Flatten()(RepeatVector(n1)(mat2))
    result = multiply(inputs=[mat1, mat2])

    # convert (i-1)dim to i dim
    # result = Reshape((n2, n1))(result)
    return result


Kronecker = Lambda(lambda tensors: kronecker_product(tensors[0], tensors[1]))

texts_images_kron = Kronecker([images_out, texts_out])
texts_leaves_kron = Kronecker([leaves_out, texts_out])
images_leaves_kron = Kronecker([images_out, leaves_out])
images_leaves_texts_kron = Kronecker([images_out, texts_leaves_kron])


if is_fusion:
    datas = [texts_out, images_out, leaves_out, texts_images_kron,
             texts_leaves_kron, images_leaves_kron, images_leaves_texts_kron]
else:
    datas = [texts_out, images_out, leaves_out]


cat_data = concatenate(datas)
# cat_hidden = Dropout(0.5)(cat_data)

cat_hidden = Dense(700, activation="relu")(cat_data)
cat_hidden = Dropout(0.5)(cat_hidden)
cat_hidden = Dense(700, activation="relu")(cat_hidden)
cat_hidden = Dropout(0.5)(cat_hidden)
cat_out = Dense(2, activation='sigmoid', name='cat_out',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                bias_regularizer=l1_l2(l1=0.01, l2=0.01),
                activity_regularizer=l1_l2(l1=0.01, l2=0.01))(cat_hidden)

model = Model(inputs=[texts, images, leaves], outputs=[cat_out])

model.compile(optimizer='adam',
              loss='binary_crossentropy', metrics=['accuracy'])

texts_ = read("/home/ygy/concat_sequences.pkl")
old_data = read("/home/ygy/features.pkl")
target_ = read("/home/ygy/label.pkl")


# images_ = old_data.iloc[:, 200:204]
images_ = read('/home/ygy/images_fixed.pkl')
leaves_ = old_data.iloc[:, 204:]

index = set(texts_.index) & set(images_.index) & set(leaves_.index) & set(target_.index)

target_ = target_.loc[~target_.index.duplicated(), :]
target_ = target_.loc[index, :]


pos_labels = target_.loc[target_.iloc[:, 0] == 1, :].iloc[:n_pos_samples, 0]
neg_labels = target_.loc[target_.iloc[:, 0] == 0, :]
labels = concat((pos_labels, neg_labels), axis=0)

train_labels, test_labels = train_test_split(labels, stratify=labels.values.ravel(), test_size=0.25)
train_index, test_index = train_labels.index, test_labels.index

texts_ = texts_.loc[~texts_.index.duplicated(), :]
train_texts = texts_.loc[train_index, :].values
test_texts = texts_.loc[test_index, :].values

images_ = images_.loc[~images_.index.duplicated(), :]
train_images = images_.loc[train_index, :].values
test_images = images_.loc[test_index, :].values

leaves_ = leaves_.loc[~leaves_.index.duplicated(), :]
train_leaves = leaves_.loc[train_index, :].values
test_leaves = leaves_.loc[test_index, :].values

train_target = np_utils.to_categorical(target_.loc[train_index, :].values)
test_target = np_utils.to_categorical(target_.loc[test_index, :].values)

# And trained it via:
early_stopping = EarlyStopping(monitor="val_acc", verbose=2, patience=5, mode="max")

model.fit(x={'texts': train_texts, 'images': train_images, 'leaves': train_leaves},
          y={'cat_out': train_target},
          validation_split=0.3,
          # validation_data=({'texts': test_texts, 'images': test_images, 'leaves': test_leaves},
          #                  {'cat_out': test_target}),
          shuffle=True,
          callbacks=[early_stopping],
          epochs=50, batch_size=32)

pred = argmax(model.predict(x={'texts': test_texts, 'images': test_images, 'leaves': test_leaves}), axis=1)

accuracy_score(y_true=argmax(test_target, axis=1),
               y_pred=pred)

classification_report(y_true=argmax(test_target, axis=1), y_pred=pred)
confusion_matrix(y_true=argmax(test_target, axis=1), y_pred=pred)