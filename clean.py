from pathlib import Path
import shutil
import os

class Clean:
    def __init__(self, filename, limit, num, result_directory):
        self.__result_directory = result_directory
        self.__result_file = f'{Path(result_directory).as_posix()}/weights/best.pt'
        self.__new_place_result_file = f'models/{filename}.pt'
        self.__train_directory = 'imgs/train'
        self.__val_directory = 'imgs/val'
        self.__train_limit = limit
        self.__num = num


    def clean(self):
        directory = Path(self.__result_directory)

        for i in range(1, self.__num + 1):
            img_filename = f'{i}.jpg'
            txt_filename = f'{i}.txt'

            target_directory = self.__train_directory if i <= self.__train_limit else self.__val_directory

            for filename in (img_filename, txt_filename):
                os.remove(f'{target_directory}/{filename}')

        for classes_directory in (self.__train_directory, self.__val_directory):
            os.remove(f'{classes_directory}/classes.txt')

        if directory.exists() and directory.is_dir():
            os.rename(self.__result_file, self.__new_place_result_file)
            shutil.rmtree(directory)
            os.remove('imgs/train.cache')
            os.remove('imgs/val.cache')
