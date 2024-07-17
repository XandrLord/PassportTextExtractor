import os
import cv2
import shutil
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Глобальные переменные
data = {}
bins = np.arange(0, 181, 5)

def calculate_angle_and_centers(src_pts: np.ndarray) -> tuple:
    """
    Вычисляет угол и центры сторон прямоугольника.

    :param src_pts: Координаты углов прямоугольника.
    :return: Угол поворота, центры двух сторон прямоугольника.
    """
    def midpoint(pt1, pt2):
        return (pt1 + pt2) / 2

    sides = [
        (src_pts[0], src_pts[1]),
        (src_pts[1], src_pts[2]),
        (src_pts[2], src_pts[3]),
        (src_pts[3], src_pts[0])
    ]

    lengths_and_centers = [
        (np.linalg.norm(side[1] - side[0]), midpoint(side[0], side[1])) for side in sides
    ]

    lengths_and_centers.sort(key=lambda x: x[0])

    _, center1 = lengths_and_centers[0]
    _, center2 = lengths_and_centers[1]

    delta = center2 - center1
    angle = np.arctan2(delta[1], delta[0]) * (180 / np.pi)

    if angle < 0:
        angle += 180

    return angle, center1, center2

def recalculate_dst_pts(src_pts: np.ndarray, image_shape: tuple) -> tuple:
    """
    Пересчитывает координаты точек назначения и размеры изображения.

    :param src_pts: Координаты углов исходного прямоугольника.
    :param image_shape: Размеры исходного изображения.
    :return: Координаты точек назначения и новые размеры изображения.
    """
    width, height = image_shape[1], image_shape[0]
    src_width = np.linalg.norm(src_pts[0] - src_pts[1])
    src_height = np.linalg.norm(src_pts[1] - src_pts[2])
    aspect_ratio = src_width / src_height
    if aspect_ratio > 1:
        new_width = width
        new_height = int(width / aspect_ratio)
    else:
        new_height = height
        new_width = int(height * aspect_ratio)
    dst_pts = np.array([[0, 0], [new_width - 1, 0], [new_width - 1, new_height - 1], [0, new_height - 1]], dtype='float32')
    return dst_pts, (new_width, new_height)

def process_image(image_path: str, base_dir: str, output_dir: str) -> None:
    """
    Обрабатывает изображение, выполняя его трансформацию и распознавание текста.

    :param image_path: Путь к изображению.
    :param base_dir: Базовая директория для хранения данных.
    :param output_dir: Директория для сохранения выходных данных.
    """
    experiment = os.path.splitext(os.path.basename(image_path))[0]
    image_output_dir = os.path.join(output_dir, experiment)
    os.makedirs(image_output_dir, exist_ok=True)

    model = YOLO('best_pass.pt')
    model.predict(source=image_path, show=False, conf=0.5, save=True, save_txt=True, name=experiment)

    try:
        coords_path = os.path.join(base_dir, 'segment', experiment, 'labels', f'{experiment}.txt')
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        with open(coords_path, 'r') as file:
            lines = file.readlines()

        text = ''
        counter = -1
        for idx, line in enumerate(lines):
            counter += 1
            parts = line.strip().split(' ')
            coords = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
            coords_array = np.array(coords, dtype=np.float32)
            renormalized_coords = coords_array * np.array([width, height])
            sums = renormalized_coords.sum(axis=1)
            diffs = renormalized_coords[:, 0] - renormalized_coords[:, 1]
            top_left = renormalized_coords[np.argmin(sums)]
            bottom_right = renormalized_coords[np.argmax(sums)]
            top_right = renormalized_coords[np.argmax(diffs)]
            bottom_left = renormalized_coords[np.argmin(diffs)]
            src_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

            dst_pts, new_size = recalculate_dst_pts(src_pts, image.shape)

            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            transformed_image = cv2.warpPerspective(image, M, new_size)

            model_txt = YOLO('best_txt.pt')
            model_txt.predict(source=transformed_image, show=False, conf=0.3, save=True, save_txt=True, name=experiment + '_txt_' + str(counter))

            try:
                txt_coords_path = os.path.join(base_dir, 'segment', experiment + '_txt_' + str(counter), 'labels', 'image0.txt')
                height_txt, width_txt = transformed_image.shape[:2]

                with open(txt_coords_path, 'r') as file_txt:
                    lines_txt = file_txt.readlines()

                angles = []
                for idx_txt, line_txt in enumerate(lines_txt):
                    parts_txt = line_txt.strip().split(' ')
                    coords_txt = [(float(parts_txt[i]), float(parts_txt[i + 1])) for i in range(1, len(parts_txt), 2)]
                    coords_array_txt = np.array(coords_txt, dtype=np.float32)
                    renormalized_coords_txt = coords_array_txt * np.array([width_txt, height_txt])
                    sums_txt = renormalized_coords_txt.sum(axis=1)
                    diffs_txt = renormalized_coords_txt[:, 0] - renormalized_coords_txt[:, 1]
                    top_left_txt = renormalized_coords_txt[np.argmin(sums_txt)]
                    bottom_right_txt = renormalized_coords_txt[np.argmax(sums_txt)]
                    top_right_txt = renormalized_coords_txt[np.argmax(diffs_txt)]
                    bottom_left_txt = renormalized_coords_txt[np.argmin(diffs_txt)]
                    src_pts_txt = np.array([top_left_txt, top_right_txt, bottom_right_txt, bottom_left_txt], dtype='float32')

                    angle, center1, center2 = calculate_angle_and_centers(src_pts_txt)

                    angles.append(angle)

                angles = np.array(angles)

                group_indices = np.digitize(angles, bins) - 1
                groups = [angles[group_indices == i] for i in range(len(bins) - 1)]

                largest_group = max(groups, key=len)

                average_angle = np.mean(largest_group)

                transformed_image = rotate_all(transformed_image, -average_angle)

                shutil.rmtree(os.path.join(base_dir, 'segment', experiment + '_txt_' + str(counter)))

                model_txt = YOLO('best_ud.pt')
                model_txt.predict(source=transformed_image, show=False, conf=0.3, save=True, save_txt=True, name=experiment + '_ud_' + str(counter))

                try:
                    ud_txt = os.path.join(base_dir, 'segment', experiment + '_ud_' + str(counter), 'labels', 'image0.txt')

                    with open(ud_txt, 'r') as file_ud:
                        lines_ud = file_ud.readlines()

                    up_count = 0
                    down_count = 0
                    for idx_ud, line_ud in enumerate(lines_ud):
                        classn = line_ud[0]

                        if classn == '0':
                            up_count += 1
                        else:
                            down_count += 1

                    if down_count > up_count:
                        transformed_image = rotate_all(transformed_image, 180)

                    shutil.rmtree(os.path.join(base_dir, 'segment', experiment + '_ud_' + str(counter)))

                    output_path_transformed = os.path.join(image_output_dir, f'{experiment}_profile_view_{idx}.png')
                    cv2.imwrite(output_path_transformed, transformed_image)

                    text += recognize_text_with_paddleocr(transformed_image)
                    text += '           '

                except Exception as e:
                    print(e)
                    shutil.rmtree(os.path.join(base_dir, 'segment', experiment + '_ud_' + str(counter)))
                    output_path_transformed = os.path.join(image_output_dir, f'{experiment}_profile_view_{idx}.png')
                    cv2.imwrite(output_path_transformed, transformed_image)
                    data[os.path.basename(image_path)] = {
                        'original_name': os.path.basename(image_path),
                        'transformed_image_path': output_path_transformed,
                    }

            except Exception as e:
                print(e)
                shutil.rmtree(os.path.join(base_dir, 'segment', experiment + '_txt_' + str(counter)))
                output_path_transformed = os.path.join(image_output_dir, f'{experiment}_profile_view_{idx}.png')
                cv2.imwrite(output_path_transformed, transformed_image)
                data[os.path.basename(image_path)] = {
                    'original_name': os.path.basename(image_path),
                    'transformed_image_path': output_path_transformed,
                }

        data[os.path.basename(image_path)] = {
            'original_name': os.path.basename(image_path),
            'transformed_image_path': output_path_transformed,
            'text': text[:-11]
        }

    except Exception as e:
        print(f'{image_path} error: {e}')
        data[os.path.basename(image_path)] = {
            'original_name': os.path.basename(image_path),
            'transformed_image_path': '',
        }

    # Очистка папки segment, кроме папок trainN
    clean_segment_folder(base_dir)

def clean_segment_folder(base_dir: str):
    """
    Очищает папку segment, кроме папок, начинающихся с 'train'.

    :param base_dir: Базовая директория.
    """
    segment_path = os.path.join(base_dir, 'segment')
    for item in os.listdir(segment_path):
        item_path = os.path.join(segment_path, item)
        if os.path.isdir(item_path) and not item.startswith('train'):
            shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            os.remove(item_path)

def process_images_in_folder(folder_path: str, base_dir: str, output_dir: str) -> None:
    """
    Обрабатывает все изображения в указанной папке.

    :param folder_path: Путь к папке с изображениями.
    :param base_dir: Базовая директория для хранения данных.
    :param output_dir: Директория для сохранения выходных данных.
    """
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        if image_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(folder_path, image_file)
            process_image(image_path, base_dir, output_dir)

def rotate_all(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Поворачивает изображение на заданный угол.

    :param img: Исходное изображение.
    :param angle: Угол поворота.
    :return: Повернутое изображение.
    """
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(rows * abs_sin + cols * abs_cos)
    new_h = int(rows * abs_cos + cols * abs_sin)
    M[0, 2] += new_w / 2 - cols / 2
    M[1, 2] += new_h / 2 - rows / 2
    dst = cv2.warpAffine(img, M, (new_w, new_h))
    return dst

def recognize_text_with_paddleocr(image: np.ndarray) -> str:
    """
    Распознает текст на изображении с помощью PaddleOCR.

    :param image: Исходное изображение.
    :return: Распознанный текст.
    """
    prep_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    results = ocr.ocr(prep_img, cls=True)

    text_res = ''

    for line in results:
        for element in line:
            bbox, (text, prob) = element[0], element[1]
            if prob > 0.7:
                text_res += text + ' '

    text_res = text_res.strip()
    return text_res