from time import sleep
import vosk
from scipy.io import wavfile
from vosk import KaldiRecognizer
import json
import wave
from nn import start_nn, train_and_return_model
from redis_scripts import CustomRedis


def cleaning_dir(path_list):
    import os

    for file in path_list:
        try:
            os.remove(file)
        except:
            pass


def recognize(path_to_file, model):
    samplerate, data = wavfile.read(path_to_file)
    wf = wave.open(fr'{path_to_file}', "rb")
    rec = KaldiRecognizer(model, samplerate)

    result = ''
    last_n = False

    while True:
        data = wf.readframes(samplerate)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())

            if res['text'] != '':
                result += f" {res['text']}"
                last_n = False
            elif not last_n:
                result += '\n'
                last_n = True

    res = json.loads(rec.FinalResult())
    result += f" {res['text']}"

    return result


def start_vosk_recognize(path_to_file: list, model) -> list:
    return [recognize(path_file, model) for path_file in path_to_file]


def write_client_phrase(recognize_result: list):
    path = 'C:\\keras\\lstm_network\\recognized_clients_phrases\\recognized_phrases.txt'

    with open(path, 'r', encoding='utf-8') as file:
        phrase_list = [phrases.replace('\n', '') for phrases in file.readlines()]

    for phrase in recognize_result:
        phrase = phrase.strip()

        if len(phrase) > 3:
            if phrase not in phrase_list:
                with open(path, 'a', encoding='utf-8') as file:
                    file.write(phrase + '\n')


def speech_recognize(file_list):
    import speech_recognition as SP
    from speech_recognition import UnknownValueError

    res_list = []

    for file in file_list:
        sample_audio = SP.WavFile(file)
        recognizer = SP.Recognizer()

        with sample_audio as audio_file:
            recognizer.adjust_for_ambient_noise(audio_file)
            audio_content = recognizer.record(audio_file)
            try:
                res_list.append(recognizer.recognize_google(audio_content, language="ru-RU"))
            except UnknownValueError:
                return 'Тишина'
    return res_list


def run_vosk():
    from threading import Thread

    def load_model():
        model = vosk.Model(model_path="C:\\vosk\\vosk-model-ru-0.10")
        print('---> Модель распознавания речи загружена')
        bd = CustomRedis()
        print('---> Подключился к бд')
        # nn_model = train_and_return_model(load=True)
        # print('---> Модель нейросети загружена')
        print('---> Готов к работе')

        while True:
            res_dict = bd.get_all()
            if len(res_dict):
                key = list(res_dict.keys())[0]
                bd_data = bd.get(key)
                path_list = bd_data if isinstance(bd_data, list) else [bd_data]

                try:
                    print(f'Начал распознование списка --> {path_list[0]}')
                    recognize_result = start_vosk_recognize(path_list, model)
                    bd.delete(key)
                    recognize_result = [res.strip() for res in recognize_result if res != ' ' or len(res) > 1]
                    print(recognize_result)
                    # start_nn(recognize_result, nn_model=nn_model)
                    write_client_phrase(recognize_result)
                    cleaning_dir(path_list)
                except:
                    print(path_list)
                    bd.delete(key)
                    cleaning_dir(path_list)

            sleep(1)
    Thread(target=load_model, name='load_model').start()
