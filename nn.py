import os
from keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.layers import Dense, LSTM, Dropout, Embedding, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


with open('C:\\keras\\lstm_network\\for_waves_scripts\\complete_datasets\\operator.txt', 'r', encoding='utf-8') as f:
    texts_operator = f.readlines()
    texts_operator = [i.replace('\ufeff', '').replace('\n', '') for i in texts_operator if i != '\\n']

with open('C:\\keras\\lstm_network\\for_waves_scripts\\complete_datasets\\client.txt', 'r', encoding='utf-8') as f:
    texts_client = f.readlines()
    texts_client = [i.replace('\ufeff', '').replace('\n', '') for i in texts_client if i != '\\n']

texts = texts_operator + texts_client

maxWordsCount = 10000
max_text_len = 100

tokenizer = Tokenizer(filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts)


def write_phrase(phrase, pred):
    exception_list = ['тишина', 'здравствуйте', 'привет', 'я', 'ты', 'вы', 'а', 'что', 'мне', 'да', 'алло']
    phrase = phrase.strip()
    if len(phrase.split(' ')) > 1:
        if (phrase.lower() not in exception_list) and len(phrase):
            if pred[0] > 0.01:
                with open('C:\\keras\\lstm_network\\for_waves_scripts\\operator.txt', 'a', encoding='utf-8') as file:
                    file.write(phrase + '\n')
            else:
                with open('C:\\keras\\lstm_network\\for_waves_scripts\\client.txt', 'a', encoding='utf-8') as file:
                    file.write(phrase + '\n')


def convert_text_tokenizer(text=None, maxlen=max_text_len):
    if text is None:
        text = texts
    else:
        text = [text]

    text_sequences = tokenizer.texts_to_sequences(text)

    return pad_sequences(text_sequences, maxlen=maxlen)


def train_and_return_model(load=False):
    model_name = 'client_operator_model'
    path_to_model_dir = f'C:\\keras\\lstm_network\\for_waves_scripts\\trained_models\\{model_name}'

    if load is False:
        X = convert_text_tokenizer()
        Y = np.array([1] * len(texts_operator) + [0] * len(texts_client))

        indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
        X = X[indeces]
        Y = Y[indeces]

        model = Sequential()
        model.add(Embedding(maxWordsCount, 256, input_length=max_text_len))
        model.add(LSTM(256, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(0.001))

        early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1, min_delta=0.001, restore_best_weights=True)

        history = model.fit(X, Y, batch_size=32, epochs=500, callbacks=[early_stop])

        model.save(path_to_model_dir)
    else:
        model = load_model(path_to_model_dir)
    return model


def start_nn(phrase_list, nn_model=None):
    if nn_model is None:
        nn_model = train_and_return_model(load=True)

    for phrase in phrase_list:
        text_for_predict = convert_text_tokenizer(text=phrase)
        pred = nn_model.predict(text_for_predict)
        write_phrase(phrase, pred)
        print(phrase, pred)


if __name__ == '__main__':
    nn_model = train_and_return_model(load=False)

    text_list = [
        'черт я получил вот я даже ещё по бритни не видела папа открытие расчётного счета',
        'не зарегистрирован между пока',
        'документы полностью тянуть',
        'да нет мы уже работаем на банк нам дополнительным не требуется',
        'это уже уже уже неактуально а я уже что-то открыл да',
        'в мой работаем глобал нас устраивают',
        'да да он вообще как-то нерабочие то есть я вам возражения говорю что мне неинтересен ваш банк он не говорит о тарифах',
        'ах мы посмотрим уже там же будут контактах для обратных звонков',
        'можете сразу напротив воротам почту',
        'лучше направить информацию',
        'можете на почту отправить предложение контактные данные',
        'контактные данные'
    ]

    for phrase in text_list:
        text_for_predict = convert_text_tokenizer(text=phrase)

        pred = nn_model.predict(text_for_predict)

        print(pred)
        if pred[0] > 0.01:
            print(f'{phrase} --> Оператор')
        else:
            print(f'{phrase} --> Клиент')
        print()

        # write_phrase(phrase, pred)
