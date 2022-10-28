import heapq
from unittest import TestCase
from for_waves_scripts.splite_waves_for_ml import *


class SoloFunctionTestCase(TestCase):
    def setUp(self) -> None:
        # self.name_without_format = 'recording_192_168_0_22_73432248066_1665743581_269_2022_10_14_17'
        # self.name_without_format = 'recording_192_168_0_22_73432248066_1665755653_1680_2022_10_14_20'
        self.name_without_format = 'recording_192_168_0_22_73432248066_1665742821_192_2022_10_14_17'

    def test_get_list(self):
        wav_fname = f'C:\keras\lstm_network\converted_waves\{self.name_without_format}.wav'
        samplerate, data = wavfile.read(wav_fname)
        # print(data)
        new_data = [digit * -1 if digit < 0 else digit for digit in data]
        print(max(new_data))
        print(min(new_data))
        new_data = list(set(new_data))
        print(heapq.nsmallest(50, new_data, key=None))

    def test_create_seconds_dicts(self):
        res = create_seconds_dicts(self.name_without_format)
        # print(res)
        test_list = []
        for key, value in res.items():
            if len(value):
                # print(key, max(value))
                test_list.append(max(value))

        # print(min(test_list))
        # print(max(test_list))
        # mean_digit = sum(test_list)/len(test_list)
        # print(mean_digit)
        # new_list = [i for i in test_list if i < mean_digit]
        #
        # min_mean_digit = sum(new_list)/len(new_list)
        # print(int(min_mean_digit))

    # def test_get_min_mean_digit(self):
    #     seconds_dicts = create_seconds_dicts(self.name_without_format)
    #     res = get_min_mean_digit(seconds_dicts)
    #     print(res)

    def test_create_seconds_dicts_two(self):
        test_res = create_seconds_dicts(self.name_without_format)
        print(test_res)

    def test_find_min_maximum(self):
        res = find_min_maximum(self.name_without_format)
        print(res)

    def test_convert_and_move(self):
        convert_and_move(self.name_without_format)

    def test_start(self):
        start(self.name_without_format)

    # def test_add_files(self):
    #     add_files()

    def test_rec(self):
        def speech_recognize(file):
            import speech_recognition as SP
            from speech_recognition import UnknownValueError

            sample_audio = SP.WavFile(file)
            recognizer = SP.Recognizer()

            with sample_audio as audio_file:
                # audio_content = recognizer.record(audio_file)
                recognizer.adjust_for_ambient_noise(audio_file)
                audio_content = recognizer.record(audio_file)
                #
                # new_path = 'C:\\keras\\lstm_network\\converted_waves\\new_2_canal\\rem_noise.wav'
                # with open(new_path, "wb") as f:
                #     f.write(audio_content.get_wav_data())
                try:
                    return recognizer.recognize_google(audio_content, language="ru-RU")
                except UnknownValueError:
                    return 'Тишина'

        file = 'C:\\keras\\lstm_network\\converted_waves\\new_2_canal\\recording_192_168_0_22_73432248066_1665755653_1680_2022_10_14_20.wav'

        res = speech_recognize(file)
        print(res)
        # newAudio = AudioSegment.from_wav(f"C:\keras\lstm_network\converted_waves\{name}.{format}")
        #
        # path_to_new_file = f'C:\keras\lstm_network\converted_waves\\sliced_waves\\{name}{t1}-{t2}.{format}'
        # newAudio.export(path_to_new_file, format=format)

    def test_get_slice_from_dict(self):
        current_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'r'}
        index = 2
        tes_res = get_slice_from_dict(current_dict, index)
        print(tes_res)
