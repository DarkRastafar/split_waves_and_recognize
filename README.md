Чтоб заюзать нейросеть по распознанию фраз клиента и оператора:
nn.py
  - 99 строка
  или
  - nn_model = train_and_return_model(load=False)

    text_list = [] - сюда пишем фразы для распознания (либо можно взять из этого же файла)

    for phrase in text_list:
        text_for_predict = convert_text_tokenizer(text=phrase)

        pred = nn_model.predict(text_for_predict)

        print(pred)
        if pred[0] > 0.01:
            print(f'{phrase} --> Оператор')
        else:
            print(f'{phrase} --> Клиент')
        print()
Чтобы перевести голос в текст:
  - запустить redis server
  splite_waves_for_ml.py
    - залить waves по пути 183 строки (или прописать новый)
    - триггернуть if __name__ == '__main__'
  vosk_main.py
    - Переписать пути, либо создать пути, взяв целевое содержимое из папок репы.
    - раскомментить 97, 98, 114 (если есть желание сразу классифицировать фразы)
    - 89 срока заюзать функцию (запишет результаты по пути (указан на 55 строке))

    
