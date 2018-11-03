from keras.models import load_model, Model
from numpy.random import standard_normal
from numpy import concatenate as np_concat, ones, zeros
from sklearn.externals import joblib
from sklearn.preprocessing import minmax_scale
from keras.utils import np_utils
from random import sample
from keras.layers import (Input, Embedding, Dense, Conv1D, MaxPool1D, Flatten,
                          Dropout, concatenate, multiply, RepeatVector, Lambda)
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping


n_gen_samples = 10000
n_true_samples = 10000
n_test = 3000


text_noise_len = 410

embedding_len = 300

image_noise_len = 50
leaf_noise_len = 350

generator_for_image = load_model("/home/ygy/gan_model/generator_for_image.h5")
generator_for_leaf = load_model("/home/ygy/gan_model/generator_for_leaf.h5")
generator_for_text = load_model("/home/ygy/gan_model/generator_for_text.h5")

text_seed_noises = standard_normal(size=(n_gen_samples,
                                         text_noise_len,
                                         embedding_len)).astype(dtype="float32")
image_seed_noises = standard_normal(size=(n_gen_samples,
                                          image_noise_len)).astype(dtype="float32")
leaf_seed_noises = standard_normal(size=(n_gen_samples,
                                         leaf_noise_len)).astype(dtype="float32")

pos_gen_embedding_mat = generator_for_text.predict(text_seed_noises)
pos_gen_images = generator_for_image.predict(image_seed_noises)
pos_gen_leaves = generator_for_leaf.predict(leaf_seed_noises)


def read(fname):
    with open(fname, 'rb') as fr:
        return joblib.load(fr)


class Word2Embedded(object):
    def __init__(self, text_len, embedding_matrix):
        text_input = Input(shape=(text_len, ), dtype="int32")
        embedded = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                             input_length=text_len, weights=[embedding_matrix], trainable=False)(text_input)
        self.model = Model(inputs=[text_input], outputs=[embedded])

    def __call__(self, texts):
        return self.model.predict(texts)


texts_ = read("/home/ygy/concat_sequences.pkl")
old_data = read("/home/ygy/features.pkl")
target_ = read("/home/ygy/label.pkl")

# images_ = old_data.iloc[:, 200:204]
images_ = read('/home/ygy/images_fixed.pkl')
leaves_ = old_data.iloc[:, 204:]

index = set(texts_.index) & set(images_.index) & set(leaves_.index) & set(target_.index)

target_ = target_.loc[~target_.index.duplicated(), :]
target_ = target_.loc[index, :]

pos_index = sample(target_.index[target_.iloc[:, 0] == 1].values.tolist(), n_true_samples + n_test)
train_pos_index = pos_index[:n_true_samples]
test_pos_index = pos_index[n_true_samples:]

neg_index = target_.index[target_.iloc[:, 0] == 0].values.tolist()
train_neg_index = neg_index[n_test:]
test_neg_index = neg_index[:n_test]

test_index = test_pos_index + test_neg_index

texts_ = texts_.loc[~texts_.index.duplicated(), :]
pos_true_texts = texts_.loc[train_pos_index, :].values
neg_true_texts = texts_.loc[train_neg_index, :].values

test_texts = texts_.loc[test_index, :].values


embedding_matrix_ = read("/home/ygy/embedding_matrix.pkl").astype("float32")
word2embedded = Word2Embedded(400, embedding_matrix_)
pos_true_embedding_mat = word2embedded(pos_true_texts)
neg_true_embedding_mat = word2embedded(neg_true_texts)

train_texts = np_concat((pos_true_embedding_mat, pos_gen_embedding_mat, neg_true_embedding_mat), axis=0)
test_texts = word2embedded(test_texts)


images_ = images_.loc[~images_.index.duplicated(), :]
pos_true_images = minmax_scale(images_.loc[train_pos_index, :].values, feature_range=(-1, 1))
neg_true_images = minmax_scale(images_.loc[train_neg_index, :].values, feature_range=(-1, 1))

train_images = np_concat((pos_true_images, pos_gen_images, neg_true_images), axis=0)
test_images = minmax_scale(images_.loc[test_index, :].values, feature_range=(-1, 1))

leaves_ = leaves_.loc[~leaves_.index.duplicated(), :]
pos_true_leaves = minmax_scale(leaves_.loc[train_pos_index, :].values, feature_range=(-1, 1))
neg_true_leaves = minmax_scale(leaves_.loc[train_neg_index, :].values, feature_range=(-1, 1))

train_leaves = np_concat((pos_true_leaves, pos_gen_leaves, neg_true_leaves), axis=0)
test_leaves = minmax_scale(leaves_.loc[test_index, :].values, feature_range=(-1, 1))

train_target = np_concat((ones(shape=n_true_samples + n_gen_samples, dtype="float32"),
                          target_.loc[train_neg_index, :].values.ravel()), axis=0)
train_target = np_utils.to_categorical(train_target)

test_target = np_concat((ones(n_test, dtype="float32"), zeros(n_test, dtype="float32")), axis=0)
test_target = np_utils.to_categorical(test_target)

# =======================================================================================================================
# build the model

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


texts = Input(shape=(max_len, embedding_len), dtype='float32', name='texts')
conv_out = Conv1D(n_filter, kernel_size, padding='valid', activation='relu', strides=1)(texts)

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

# And trained it via:
early_stopping = EarlyStopping(monitor="val_acc", verbose=2, patience=5, mode="max")

model.fit(x={'texts': train_texts, 'images': train_images, 'leaves': train_leaves},
          y={'cat_out': train_target},
          # validation_split=0.3,
          validation_data=({'texts': test_texts, 'images': test_images, 'leaves': test_leaves},
                           {'cat_out': test_target}),
          shuffle=True,
          callbacks=[early_stopping],
          epochs=50, batch_size=32)
