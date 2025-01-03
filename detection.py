from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from time import time
import cv2
import torch

class Detection:
    def __init__(self, cap_param):
        # load model
        self.__model = YOLO('yolov8n-face.pt')

        # file
        self.__directory = 'imgs/temp'
        self.__num = 0

        # image store time
        self.__start_time = 0
        self.__duration_video = 30

        # start time
        self.__time = 0
        self.__duration_save_photo = 0.1

        # capture parameter
        self.__cap_param = cap_param
        self.__min_conf = 0.75
    
    def get_num(self):
        return self.__num
    
    def get_result_directory(self):
        return self.__model.trainer.save_dir
    
    def set_cap_param(self, cap_param):
        self.__cap_param = cap_param
    
    def __plot(self, frame, boxes, cls, conf, names):

        annotator = Annotator(frame)

        for box, conf, cls in zip(boxes, conf, cls):
                annotator.box_label(box, f'{names[int(cls)]} {conf.item():.2f}')
    
    def __save_img(self, frame):
        img_filename = f'{self.__num}.jpg'
        img_file = f'{self.__directory}/{img_filename}'
        cv2.imwrite(img_file, frame)

    def __save_txt(self, x_norm, y_norm, width_norm, height_norm):
        txt_fiename = f'{self.__num}.txt'
        txt_file = f'{self.__directory}/{txt_fiename}'
        txt_value = f'0 {x_norm} {y_norm} {width_norm} {height_norm}'

        f = open(txt_file, 'w')
        f.write(txt_value)
        f.close()
    
    def __clean_value(self, frame, boxes, cls, conf, names):

        if boxes.size(0) == 1 and time() - self.__time >= self.__duration_save_photo and torch.all(conf >= self.__min_conf):

            # prepare value
            x_center = (boxes[:, 0] + boxes[:, 2]) / 2
            y_center = (boxes[:, 1] + boxes[:, 3]) / 2
            width = abs(boxes[:, 2] - boxes[:, 0])
            height = abs(boxes[:, 3] - boxes[:, 1])
            x_size = frame.shape[1]
            y_size = frame.shape[0]

            # normalized value
            x_norm = x_center/x_size
            y_norm = y_center/y_size
            width_norm = width/x_size
            height_norm = height/y_size

            # format with 6 decimal places
            x_norm = f'{x_norm.item():.6f}'
            y_norm = f'{y_norm.item():.6f}'
            width_norm = f'{width_norm.item():.6f}'
            height_norm = f'{height_norm.item():.6f}'

            # number for filename
            self.__num += 1

            self.__save_img(frame)

            self.__save_txt(x_norm, y_norm, width_norm, height_norm)

            self.__plot(frame, boxes, cls, conf, names)

            self.__time = time()


    
    def __detection(self, frame):
        results = self.__model(frame, verbose=False)

        boxes = results[0].boxes.xyxy.cpu()
        cls = results[0].boxes.cls.cpu()
        conf = results[0].boxes.conf.cpu()

        names = self.__model.names

        self.__clean_value(frame, boxes, cls, conf, names)


    def __processing(self):
        cap = cv2.VideoCapture(self.__cap_param)

        self.__start_time = time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            self.__detection(frame)

            cv2.imshow('frame', frame)

            # if time() - self.__start_time >= self.__duration_video:
            #     break

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.__processing()
    
    def training(self):
        self.__model.train(data='data.yaml', epochs=100)
    