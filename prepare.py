import shutil
import os
from pathlib import Path

class Prepare:
    def __init__(self, name, num):
        self.__num = num
        self.__current_path = Path(os.getcwd()).as_posix()
        self.__directory = 'imgs/temp'
        self.__train_directory = 'imgs/train'
        self.__val_directory = 'imgs/val'
        self.__train_value = 0.8
        self.__name = name
        self.__train_limit = 0
    
    def get_train_limit(self):
        return self.__train_limit

    def prepare(self):
        self.__train_limit = self.__num * self.__train_value

        yaml_value = f'''train: {self.__current_path}/{self.__train_directory}
val: {self.__current_path}/{self.__val_directory}

names:
 0: {self.__name}
'''

        for i in range(1, self.__num + 1):
            img_filename = f'{i}.jpg'
            txt_filename = f'{i}.txt'

            target_directory = self.__train_directory if i <= self.__train_limit else self.__val_directory

            for filename in (img_filename, txt_filename):
                shutil.move(f'{self.__directory}/{filename}', f'{target_directory}/{filename}')
            # shutil.move(f'{self.__directory}/{img_filename}', f'{target_directory}/{img_filename}')
        
        f = open('data.yaml', 'w')
        f.write(yaml_value)
        f.close()
        for directory in (self.__train_directory, self.__val_directory):
            f = open(f'{directory}/classes.txt', 'w')
            f.write(self.__name)
            f.close()
