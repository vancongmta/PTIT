# coding=utf-8
import imp
import sys
imp.reload(sys)
import numpy as np
import re
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from gensim.corpora.dictionary import Dictionary
import multiprocessing
from sklearn.model_selection import train_test_split
from keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Activation, Flatten, Masking, LayerNormalization, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPooling1D, Bidirectional
from gensim.models.word2vec import LineSentence
from keras import backend as K
from tensorflow.keras.layers import MultiHeadAttention, GlobalMaxPooling1D, Add, Dropout, Layer
from tensorflow.keras.models import Sequential

K.clear_session()

np.random.seed(1337)  # Để tái tạo lại
# Kích thước của vector từ
vocab_dim = 256
# Độ dài câu
maxlen = 350
# Số lần lặp
n_iterations = 1
# Số từ xuất hiện
n_exposures = 1
# Khoảng cách tối đa giữa từ hiện tại và từ dự đoán trong một câu
window_size = 20
# Kích thước batch
batch_size = 256
# Số epoch
n_epoch = 80
# Độ dài đầu vào
input_length = 350
# Số CPU đa xử lý
cpu_count = multiprocessing.cpu_count()

labels = ["safe", "CWE-78", "CWE-79", "CWE-89", "CWE-90", "CWE-91", "CWE-95", "CWE-98", "CWE-601", "CWE-862"]

def combine(safeFile, unsafeFile, unsafeYFile):
    global labels
    with open(safeFile, 'r') as f:
        safe_tokens = f.readlines()
    with open(unsafeFile, 'r') as f:
        unsafe_tokens = f.readlines()
    combined = np.concatenate((unsafe_tokens, safe_tokens))
    
    with open(unsafeYFile, 'r') as f:
        unsafe_labels = f.readlines()

    def tran_label(label):
        y_oh = np.zeros(10)
        y_oh[labels.index(label)] = 1
        return y_oh

    y = np.concatenate((np.array([tran_label(i.strip()) for i in unsafe_labels]), 
                        np.array([tran_label("safe") for i in safe_tokens])))
    return combined, y

def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.key_to_index.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        w2vec = {word: model.wv[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except KeyError:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')

def word2vec_train(combined):
    model = Word2Vec(vector_size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     epochs=n_iterations)
    sentences = LineSentence('./traindata_x1.txt')
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=50)
    model.save('./Word2vec_model.pkl')

    data = []
    for sentence in combined:
        words = sentence.split()
        data.append(words)

    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=data)
    return index_dict, word_vectors, combined

def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_validate, y_train, y_validate = train_test_split(combined, y, test_size=0.125)
    return n_symbols, embedding_weights, x_train, y_train, x_validate, y_validate

# Định nghĩa lớp Transformer Encoder
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(n_symbols, embedding_weights):
    embed_dim = vocab_dim  # Dữ liệu embedding size
    num_heads = 8
    ff_dim = 256

    inputs = Input(shape=(input_length,))
    embedding_layer = Embedding(input_dim=n_symbols,
                                output_dim=embed_dim,
                                weights=[embedding_weights],
                                input_length=input_length,
                                trainable=False)(inputs)
    
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(embedding_layer)
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    print('Compiling the Model...')
    adam = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model

def train_transformer(n_symbols, embedding_weights, x_train, y_train, x_validate, y_validate):
    print('Defining the Transformer Model...')
    model = build_transformer_model(n_symbols, embedding_weights)
    
    print("Training...")
    class_weight = {i: 1 for i in range(10)}  # Bạn có thể điều chỉnh trọng số lớp nếu cần
    
    model.fit(x_train, y_train, 
              batch_size=batch_size, 
              epochs=n_epoch, 
              verbose=2, 
              shuffle=True,
              class_weight=class_weight, 
              validation_data=(x_validate, y_validate))
    
    print("Evaluating...")
    score = model.evaluate(x_validate, y_validate, batch_size=batch_size)
    
    model_json = model.to_json()
    with open('./transformer_model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('./transformer.h5')
    print('Test score:', score)

def train():
    print('Token hóa...')
    safeFile = './safenew.txt'
    unsafeFile = './unsafenew.txt'
    unsafeYFile = './unsafe_y.txt'
    combined_x, combined_y = combine(safeFile, unsafeFile, unsafeYFile)

    x_train_validate, x_test, y_train_validate, y_test = train_test_split(combined_x, combined_y, test_size=0.2)

    with open('./testdata_x1.txt', 'w') as f:
        for i in x_test:
            f.write(i)
    with open('./testdata_y1.txt', 'w') as f:
        for i in y_test:
            f.write(str(i))
            f.write('\n')
    with open('./traindata_x1.txt', 'w') as f:
        for i in x_train_validate:
            f.write(i)
    with open('./traindata_y1.txt', 'w') as f:
        for i in y_train_validate:
            f.write(str(i))
            f.write('\n')

    print('Tổng cộng: ', len(x_train_validate) + len(x_test), len(y_train_validate) + len(y_test))
    print('Huấn luyện và Xác thực:', len(x_train_validate), len(y_train_validate))
    print('Kiểm tra: ', len(x_test), len(y_test))

    print('Huấn luyện một mô hình Word2vec...')
    index_dict, word_vectors, x_train_validate = word2vec_train(x_train_validate)

    print('Thiết lập Mảng cho Lớp Nhúng Keras...')
    n_symbols, embedding_weights, x_train, y_train, x_validate, y_validate = get_data(index_dict, word_vectors, x_train_validate, y_train_validate)
    print(x_train.shape, y_train.shape)

    data = []
    for sentence in x_test:
        words = sentence.split()
        data.append(words)
    model_w2v = Word2Vec.load('./Word2vec_model.pkl')
    _, _, x_test = create_dictionaries(model=model_w2v, combined=data)

    train_transformer(n_symbols, embedding_weights, x_train, y_train, x_validate, y_validate)

def input_transform(string):
    words = string.split()
    words = np.array(words).reshape(1, -1)
    model_w2v = Word2Vec.load('./Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model=model_w2v, combined=words)
    return combined

def transformer_predict():
    global labels
    print('Đang tải mô hình...')
    with open('./transformer_model.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)

    print('Đang tải trọng số...')
    model.load_weights('./transformer.h5')
    
    with open('./testdata_x1.txt', 'r') as f:
        strings = f.readlines()

    with open('./testdata_y1.txt', 'r') as f:
        y = f.readlines()

    i = 0
    đúng = 0
    sai = 0
    prevalue = ''
    preresult = ''
    
    for string in strings:
        data = input_transform(string)
        probabilities = model.predict(data)[0]
        result = np.argmax(probabilities)
        value = probabilities

        prevalue += (','.join(str(i) for i in value) + '\n')
        preresult += (str(result) + '\n')

        t = (1 + result * 3)
        if 1 == int(y[i][t:t+1]):
            đúng += 1
        else:
            sai += 1

        i += 1

    with open('./predict_value.txt', 'w') as f:
        f.write(prevalue)
    with open('./predict_result.txt', 'w') as f:
        f.write(preresult)

    print('Đúng: ', đúng, ' Sai: ', sai)
    print('Độ chính xác: ', đúng / (đúng + sai))
    print('Kết quả dự đoán ghi vào file predict_value.txt và predict_result.txt\n')

if __name__ == '__main__':
    train()
    transformer_predict()
