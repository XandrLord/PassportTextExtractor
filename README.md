# **PassportTextExtractor: программа для получения данных с паспортов, представленных на фото**

## **Установка и подготовка**

### **Установка зависимостей**

Для установки всех необходимых зависимостей выполните следующую команду:

```
pip install -r requirements.txt
```

Если возникает ошибка - установите вручную библиотеки, представленные в файле.

### **Изменение библиотеки YOLO**

Требуется заменить файл results.py в папке engine библиотеки ultralytics на версию из папки ultrachanges.

Пример расположения:

```
C:\Users\Пользователь\PycharmProjects\SMTH\venv\Lib\site-packages\ultralytics\engine
```

## **Запуск версии программы с GUI через терминал**

```
python gui.py
```
В поле для ввода текста требуется вписать путь к системной папке YOLO.

## **Запуск версии программы с Flask через терминал**

```
python app.py
```
В поле для ввода текста требуется вписать путь к системной папке YOLO.

## **Запуск базовой версии программы через терминал**

Общий формат команды

```
python main.py <тип обработки (folder - папка, image - одно изображение)> <системная папка YOLO> <путь к данным (к папке/к фото) для обработки> <путь к папке, где будут обработанные изображения> <путь к папке для сохранения json>
```

### **Примеры использования**

Обработка одного изображения

```
python main.py image <base_dir> <image_path> <output_dir> <json_output_path>
```

Обработка всех изображений в папке

```
python main.py folder <base_dir> <folder_path> <output_dir> <json_output_path>
```

### **Конкретные примеры**

Обработка одного изображения

```
python main.py image c:\users\alexandr\runs C:\Users\Alexandr\Downloads\test1.jpg C:\Users\Alexandr\Downloads C:\Users\Alexandr\Downloads\test.json
```

Обработка всех изображений в папке

```
python main.py folder c:\users\alexandr\runs C:\Users\Alexandr\Downloads C:\Users\Alexandr\Downloads C:\Users\Alexandr\Downloads\test.json
```

## **Ограничения использования**

Код имеет ограничения на использование в связи с авторскими правами.