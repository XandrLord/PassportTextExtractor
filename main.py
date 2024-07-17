import argparse
import os
import json
from image_processor import process_image, process_images_in_folder, data

# Устанавливаем переменную окружения для предотвращения конфликта с библиотекой OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def main():
    """
    Основная функция для обработки изображений в зависимости от выбранного режима.
    """
    # Инициализация парсера аргументов командной строки
    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('mode', type=str, choices=['folder', 'image'], help='Mode of processing: "folder" or "image"')
    parser.add_argument('base_dir', type=str, help='Base directory for processing')
    parser.add_argument('path', type=str, help='Folder path or image path to process')
    parser.add_argument('output_dir', type=str, help='Directory to save the results')
    parser.add_argument('json_output_path', type=str, help='Path to save the output JSON file')

    # Парсинг аргументов командной строки
    args = parser.parse_args()

    # Обработка изображений в зависимости от выбранного режима
    if args.mode == 'folder':
        process_images_in_folder(args.path, args.base_dir, args.output_dir)
    elif args.mode == 'image':
        process_image(args.path, args.base_dir, args.output_dir)

    # Создание директорий для файла JSON, если они не существуют
    os.makedirs(os.path.dirname(args.json_output_path), exist_ok=True)

    # Сохранение результатов обработки в файл JSON
    with open(args.json_output_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

if __name__ == "__main__":
    main()