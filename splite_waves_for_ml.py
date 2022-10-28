import datetime
import heapq
import math
from scipy.io import wavfile
import scipy.io
from pydub import AudioSegment
from remove_noiz import removeNoise
from vosk_main import CustomRedis

path_to_file = 'C:\keras\lstm_network\converted_waves\david_i_tatiana.wav'


def remove_noize_main(path_to_file, noise_example):
    rate, audio_clip = wavfile.read(path_to_file)
    audio_clip = audio_clip / 32768

    rate, noise_clip = wavfile.read(noise_example)
    noise_clip = noise_clip / 32768

    removed_noize_audio = removeNoise(audio_clip=audio_clip, noise_clip=noise_clip)
    new_path = path_to_file.replace('.wav', '')
    new_path = f'{new_path}_rem_noise.wav'
    scipy.io.wavfile.write(new_path, rate, removed_noize_audio)

    # fig, ax = plt.subplots(figsize=(20, 4))
    # ax.plot(removed_noize_audio)
    # plt.show()
    return new_path


def convert_to_wave(file_name_without_type):
    file_name = file_name_without_type
    dot_type = '.mp3'

    path_to_old_file = f'C:\keras\lstm_network\mp3_files\{file_name}{dot_type}'

    AudioSegment.converter = "C:\\ffmpeg\\bin\\ffmpeg.exe"
    sound = AudioSegment.from_mp3(path_to_old_file)
    sound.export(f"C:\\keras\\lstm_network\\converted_waves\\{file_name}.wav", format="wav")
    return True


summary_digit = 2


def update_list(res_list, total_index, i):
    try:
        if i not in res_list[total_index]:
            res_list[total_index].append(i)
    except:
        res_list.append([])
        res_list[total_index].append(i)


def create_seconds_dicts(path_to_file) -> dict:
    wav_fname = f'C:\keras\lstm_network\converted_waves\{path_to_file}.wav'

    samplerate, data = wavfile.read(wav_fname)
    length = data.shape[0] / samplerate
    seconds = math.ceil(length) * summary_digit

    res_dict = {}
    mean_elem = int(len(data) / seconds)
    last_index = 0

    for i in range(seconds):
        new_index = mean_elem * i
        res_dict.update({i: [digit * -1 if digit < 0 else digit for digit in data[last_index: new_index]]})
        last_index = new_index
    return res_dict


def get_min_list(path_to_file):
    wav_fname = f'C:\keras\lstm_network\converted_waves\{path_to_file}.wav'
    samplerate, data = wavfile.read(wav_fname)
    new_data = [digit * -1 if digit < 0 else digit for digit in data]
    new_data = list(set(new_data))
    return heapq.nsmallest(50, new_data, key=None)


def get_slice_from_dict(current_dict, index) -> dict:
    return {key: value for key, value in current_dict.items() if key >= index}


def check_value(value, min_digit_list):
    for min_digit in min_digit_list:
        if float(value.count(min_digit)) > float(len(value) / 2):
            return False
    return True


def find_min_maximum(path_to_file):
    res_dict = create_seconds_dicts(path_to_file)
    min_digit_list = get_min_list(path_to_file)
    min_max_digit = max(min_digit_list)
    slices_without_silent = [[]]
    total_index = 0
    back_step_digit = 0

    def update_slices_without_silent(res_dict, total_index, back_step_digit):
        for key, value in res_dict.items():
            if len(value):
                value = max(value)
                if value > min_max_digit:
                    if back_step_digit > min_max_digit:
                        update_list(slices_without_silent, total_index, key)
                        back_step_digit = value
                    else:
                        total_index += 1
                        update_list(slices_without_silent, total_index, key)
                        back_step_digit = value
                else:
                    if back_step_digit != 0:
                        try:
                            key_list = []
                            for i in range(1, 1):
                                next_key = key + i
                                key_list.append(next_key)
                                if max(res_dict[next_key]) > min_max_digit:
                                    for lower_keys in key_list:
                                        update_list(slices_without_silent, total_index, lower_keys)
                                    return update_slices_without_silent(get_slice_from_dict(res_dict, next_key), total_index, back_step_digit)
                        except:
                            update_list(slices_without_silent, total_index, key)
                    back_step_digit = value

    update_slices_without_silent(res_dict, total_index, back_step_digit)

    return [i for i in slices_without_silent if len(i)]


def return_seconds_list(path_to_file):
    return [[min(i), max(i)] for i in find_min_maximum(path_to_file)]


def create_slice_wave(name, t1, t2, format='wav'):
    t1 = t1 * 1000 / summary_digit
    t1 = t1 - 1000 if t1 > 1000 else 0
    t2 = t2 * 1000 / summary_digit
    newAudio = AudioSegment.from_wav(f"C:\keras\lstm_network\converted_waves\{name}.{format}")
    newAudio = newAudio[t1:t2]

    path_to_new_file = f'C:\keras\lstm_network\converted_waves\\sliced_waves\\{name}{t1}-{t2}.{format}'
    newAudio.export(path_to_new_file, format=format)

    return path_to_new_file


def convert_and_move(file_name_without_type, path_to_dir='C:\\keras\\lstm_network\\mp3_files\\'):
    import shutil
    from os import path

    if convert_to_wave(file_name_without_type) is True:
        source_path = f'{path_to_dir}{file_name_without_type}.mp3'
        destination_path = "C:\\keras\\lstm_network\\complete_mp3_files"

        if path.exists(source_path):
            new_location = shutil.move(source_path, destination_path)
            print("% s перемещен в указанное место,% s" % (source_path, new_location))
            return new_location
        else:
            print("Файл не существует.")
    else:
        print(f"Не удалось конвертировать --> {path_to_dir}{file_name_without_type}.")


def start(name_without_format):
    bd = CustomRedis()
    path_list = []
    for i in return_seconds_list(name_without_format):
        path_to_wave = create_slice_wave(name_without_format, i[0], i[1])
        path_list.append(path_to_wave)

    bd.set(
        key=f'{datetime.datetime.now()}',
        value=path_list
    )


if __name__ == '__main__':
    import glob

    path_to_wav = 'C:\\keras\\lstm_network\\converted_waves\\'

    for file in glob.glob(f"{path_to_wav}*.wav"):
        name_without_format = file.replace(f'{path_to_wav}', '').replace('.wav', '')
        start(name_without_format)
        print(f'complete --> {file}')
