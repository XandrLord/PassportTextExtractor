import os
import shutil
from tkinter import Tk, Frame, Button, Label, filedialog, Scrollbar, Canvas, Entry, Text
from PIL import Image, ImageTk
import math
from image_processor import process_image_app, data, clean_folder
import json


class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")

        # Кнопки
        self.btn_select = Button(root, text="Выбрать изображение", state='disabled', command=self.select_image)
        self.btn_select.grid(row=0, column=0, padx=10, pady=10)

        self.btn_save = Button(root, text="Скачать результат", command=self.save_result)
        self.btn_save.grid(row=0, column=1, padx=10, pady=10)

        # Поле ввода текста
        self.entry = Entry(root, width=50)
        self.entry.grid(row=0, column=2, padx=10, pady=10)
        self.entry.bind("<KeyRelease>", self.check_entry)

        # Поле для вывода текста
        self.text_output = Text(root, height=30, width=50)
        self.text_output.grid(row=1, column=2, padx=10, pady=10)

        # Создаем рамки
        self.frame1 = Frame(root, width=450, height=450, bg='lightgrey')
        self.frame1.grid(row=1, column=0, padx=10, pady=10)

        self.canvas_frame = Frame(root)
        self.canvas_frame.grid(row=1, column=1, padx=10, pady=10)

        self.canvas2 = Canvas(self.canvas_frame, width=630, height=600, bg='lightgrey')
        self.canvas2.grid(row=0, column=0)

        # Вертикальный скроллбар
        self.scrollbar_y = Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas2.yview)
        self.scrollbar_y.grid(row=0, column=1, sticky='ns')

        # Горизонтальный скроллбар
        self.scrollbar_x = Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas2.xview)
        self.scrollbar_x.grid(row=1, column=0, sticky='ew')

        self.canvas2.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        self.frame2 = Frame(self.canvas2, bg='lightgrey')
        self.canvas2.create_window((0, 0), window=self.frame2, anchor='nw')

        self.frame2.bind("<Configure>", lambda e: self.canvas2.configure(scrollregion=self.canvas2.bbox("all")))

        # Метка для отображения выбранного изображения
        self.label1 = Label(self.frame1)
        self.label1.pack()

        self.image_path = None
        self.result_images = {}
        self.result_text = ''
        self.user_input = ''

    def check_entry(self, event):
        if self.entry.get().strip():
            self.btn_select.config(state='normal')
        else:
            self.btn_select.config(state='disabled')

    def select_image(self):
        # Очистка сетки и текстового поля перед загрузкой нового изображения
        self.clear_results()

        self.image_path = filedialog.askopenfilename(title="Выберите изображение",
                                                     filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.display_image(self.image_path, self.label1)
            self.user_input = self.entry.get()
            self.result_images, self.result_text, self.result_data = self.process_images(self.image_path, self.user_input)
            self.json_dir = os.path.join(self.user_input, 'passport_text_executor')

            # Сохраняем данные в формате JSON
            with open(os.path.join(os.path.join(os.path.dirname(self.user_input), 'passport_text_executor'), 'result_data.json'), 'w', encoding='utf-8') as json_file:
                json.dump(self.result_data, json_file, ensure_ascii=False, indent=4)

            self.display_images(self.result_images, self.frame2)
            self.text_output.delete(1.0, 'end')
            self.text_output.insert('end', self.result_text)
            # clean_folder(os.path.join(os.path.dirname(self.user_input), 'passport_text_executor'))

    def clear_results(self):
        # Очистка сетки
        for widget in self.frame2.winfo_children():
            widget.destroy()

        # Очистка текстового поля
        self.text_output.delete(1.0, 'end')

        # Очистка метки с изображением
        self.label1.config(image='')
        self.label1.image = None

        # Очистка сохраненных результатов
        self.result_images = {}
        self.result_text = ''

    def display_image(self, path, label):
        img = Image.open(path)
        img.thumbnail((450, 450))
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk

    def display_images(self, paths, frame):
        for widget in frame.winfo_children():
            widget.destroy()

        num_images = len(paths)
        grid_size = math.ceil(math.sqrt(num_images))

        for i, (path) in enumerate(paths):
            img = Image.open(path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            label = Label(frame, image=img_tk, compound='top')
            label.image = img_tk
            label.grid(row=i // grid_size, column=i % grid_size, padx=10, pady=10)

        frame.update_idletasks()
        self.canvas2.configure(scrollregion=self.canvas2.bbox("all"))

    def process_images(self, current_photo, base_dir):
        # Вызов process_image_app для текущего фото
        process_image_app(current_photo, base_dir)

        # Найти ключ в data, соответствующий current_photo
        photo_name = os.path.basename(current_photo)

        if photo_name in data:
            # Найти text и transformed_image_dir
            text = data[photo_name]['text']
            transformed_image_dir = data[photo_name]['transformed_image_dir']

            # Создать массив путей ко всем фото в директории transformed_image_dir
            if os.path.exists(transformed_image_dir) and os.path.isdir(transformed_image_dir):
                all_photos = [os.path.join(transformed_image_dir, file) for file in os.listdir(transformed_image_dir) if
                              os.path.isfile(os.path.join(transformed_image_dir, file))]

                return all_photos, text, data[photo_name]

            else:
                return [], text, data[photo_name]

        else:
            return [], '', {}

    def save_result(self):
        if self.result_images:
            save_dir = filedialog.askdirectory(title="Выберите папку для сохранения")
            if save_dir:
                for i, (img_path) in enumerate(self.result_images):
                    save_path = os.path.join(save_dir, os.path.basename(img_path))
                    shutil.copy(img_path, save_path)

                    # Сохраняем данные в формате JSON
                    with open(os.path.join(save_dir, 'result_data.json'), 'w', encoding='utf-8') as json_file:
                        json.dump(self.result_data, json_file, ensure_ascii=False, indent=4)

                    print(f"Результат сохранен в: {save_path}")


if __name__ == "__main__":
    root = Tk()
    app = ImageProcessor(root)
    root.mainloop()

# c:\users\alexandr\runs