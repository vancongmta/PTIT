#coding=utf-8
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
from keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Activation, Flatten, Masking
from keras.optimizers import Adam
from keras.layers import Conv1D, MaxPool1D, Bidirectional
from gensim.models.word2vec import LineSentence
from keras import backend as K

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
batch_size = 512
# Số epoch
n_epoch = 80
# Độ dài đầu vào
input_length = 350
# Số CPU đa xử lý
cpu_count = multiprocessing.cpu_count()

labels = ["safe", "CWE-78", "CWE-79", "CWE-89", "CWE-90", "CWE-91", "CWE-95", "CWE-98", "CWE-601", "CWE-862"]
def combine(safe_file, unsafe_file, unsafe_y_file):
    # Đọc dữ liệu từ các file
    with open(safe_file, 'r', encoding='utf-8') as f:
        safe_data = f.readlines()

    with open(unsafe_file, 'r', encoding='utf-8') as f:
        unsafe_data = f.readlines()

    with open(unsafe_y_file, 'r', encoding='utf-8') as f:
        unsafe_y = f.readlines()

    # Chuyển đổi các nhãn từ file unsafe_y
    unsafe_y = np.array([int(y.strip()) for y in unsafe_y])

    # Phân loại dữ liệu unsafe dựa trên nhãn
    labels = np.unique(unsafe_y)
    data_by_class = {label: [] for label in labels}

    for data, label in zip(unsafe_data, unsafe_y):
        data_by_class[label].append(data)

    # Tìm nhãn có số lượng lớn nhất ngoại trừ nhãn 'safe'
    max_count = max([len(data) for label, data in data_by_class.items() if label != 0])

    combined_x = list(safe_data)
    combined_y = [0] * len(safe_data)  # Nhãn 'safe' là 0

    # Cân bằng các nhãn khác trừ nhãn 'safe'
    for label, class_data in data_by_class.items():
        if label != 0:  # Nhãn 'safe' là 0, nên bỏ qua
            if len(class_data) < max_count:
                # Nếu số lượng nhỏ hơn, lặp lại dữ liệu
                indices = np.random.choice(len(class_data), max_count, replace=True)
                balanced_data = [class_data[i] for i in indices]
            else:
                # Nếu đủ số lượng, không cần thêm
                balanced_data = class_data

            # Thêm dữ liệu đã cân bằng vào tập combined
            combined_x.extend(balanced_data)
            combined_y.extend([label] * len(balanced_data))

    return combined_x, combined_y
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

def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_validate, y_validate):
    print('Defining an Enhanced Keras Model...')
    model = Sequential()
    
    # Lớp Embedding
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))

    # Lớp Conv1D và MaxPooling1D để cải thiện khả năng trích xuất đặc trưng
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPool1D(pool_size=4))
    
    # LSTM thông thường
    model.add(LSTM(128, return_sequences=False))
    
    # Thay đổi Dropout
    model.add(Dropout(0.4))
    
    # Lớp Dense đầu ra với softmax
    model.add(Dense(10, activation='softmax'))

    print('Compiling the Model...')
    adam = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    print("Training...")

    class_weight = {i: 1 for i in range(10)}
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=2, shuffle=True,
              class_weight=class_weight, validation_data=(x_validate, y_validate))

    print("Evaluating...")
    score = model.evaluate(x_validate, y_validate, batch_size=batch_size)
    
    model_json = model.to_json()
    with open('./lstm_model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('./lstm.h5')
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
    model = Word2Vec.load('./Word2vec_model.pkl')
    _, _, x_test = create_dictionaries(model=model, combined=data)

    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_validate, y_validate)

def input_transform(string):
    words = string.split()
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('./Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    return combined

def lstm_predict():
    global labels
    print('Đang tải mô hình...')
    with open('./lstm_model.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)

    print('Đang tải trọng số...')
    model.load_weights('./lstm.h5')
    
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
        # Dự đoán xác suất
        probabilities = model.predict(data)[0]
        # Xác định lớp dự đoán
        result = np.argmax(probabilities)
        value = probabilities

        prevalue += (','.join(str(i) for i in value) + '\n')
        preresult += (str(result) + '\n')

        # So sánh với nhãn đúng
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

if __name__ == '__main__':
    train()
    lstm_predict()
