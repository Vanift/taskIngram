import itertools
import logging
import os
import shutil
import threading
from pathlib import Path
from queue import Queue
import dataclasses
from typing import Tuple

import cv2
import numpy as np


Image = np.array
PathT = str | Path


@dataclasses.dataclass
class Vector:
    x: int
    y: int

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)


def read_image(path: str) -> Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    image = cv2.imread(path)
    logging.info(f'{path} loaded')
    return image


def read_video(path: str, queue: Queue) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    video_capture = cv2.VideoCapture(path)
    logging.info(f'{path} loaded')

    for i in itertools.count():
        ret, frame = video_capture.read()
        if not ret or frame is None or not video_capture.isOpened():
            break
        logging.debug(f'putting frame {i}')
        queue.put(frame)

    video_capture.release()


def save_image(image: Image, filename: str) -> None:
    cv2.imwrite(filename, image)
    logging.info(f'{filename!r} saved')


def find_position(frame: Image, image: Image, threshold: float = 1.0) -> Vector:
    result = cv2.matchTemplate(frame, image, cv2.TM_CCOEFF_NORMED)
    location = (result >= threshold).nonzero()
    for pt in zip(*location[::-1]):  # switch columns and rows
        logging.debug(f'image position is {pt}')
        return Vector(*pt)
    raise RuntimeError(f'no template found on frame with {threshold=}')


def draw_rectangle(image: Image, point1: Vector, point2: Vector) -> Image:
    return cv2.rectangle(image, dataclasses.astuple(point1), dataclasses.astuple(point2), (0, 0, 255), 2)


def shape(image: Image) -> Vector:
    y, x = image.shape[:2]
    return Vector(x, y)


def match_images(producer_queue: Queue[Image],
                 consumer_queue: Queue[Tuple[Image, Vector]],
                 template: Image,
                 threshold: float = 1.0):
    while True:
        frame = producer_queue.get()
        try:
            coords = find_position(frame, template, threshold)
        except RuntimeError as e:
            logging.error(e)
            continue
        new_frame = draw_rectangle(frame, coords, coords + shape(template))
        consumer_queue.put((new_frame, coords))
        logging.debug(f'frame processed ({coords=})')


def image_filename(folder: PathT, template: str, i: int, coords: Vector) -> Path:
    return Path(folder) / template.format(n=i, x=coords.x, y=coords.y)


def recreate_dir(out_dir: PathT):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)


def save_images(producer_queue: Queue, out_dir: PathT):
    recreate_dir(out_dir)

    for frame_n in itertools.count():
        image, coords = producer_queue.get()
        filename = image_filename(out_dir, '{n}_{x}_{y}.png', frame_n, coords)
        save_image(image, str(filename))
        producer_queue.task_done()


# class ThreadA(threading.Thread):
#     def __init__(self, path: str, consumer_queue: Queue, *args, **kwargs):
#         self._path = path
#         self._consumer_queue = consumer_queue
#         super().__init__(*args, **kwargs)
#
#     def run(self) -> None:
#         logging.debug(f'{self.__class__.__name__} started')
#         read_video(self._path, self._consumer_queue)


# class ThreadB(threading.Thread):
#     def __init__(self, producer_queue: Queue, consumer_queue: Queue, template: Image, threshold: float, *args, **kwargs):
#         self._producer_queue = producer_queue
#         self._consumer_queue = consumer_queue
#         self._template = template
#         self._threshold = threshold
#         super().__init__(*args, **kwargs)
#
#     def run(self):
#         logging.debug(f'{self.__class__.__name__} started')
#         return match_images(self._producer_queue, self._consumer_queue, self._template, self._threshold)


def main(video_path: str, template_path: str, out_dir: str, threshold: float):
    q1, q2 = Queue(), Queue()

    # reader = ThreadA(video_path, q1, name='reader')
    reader = threading.Thread(target=read_video, args=[video_path, q1], name='reader')

    template = read_image(template_path)
    # processor = ThreadB(q1, q2, template, threshold, name='processor')
    processor = threading.Thread(target=match_images, args=[q1, q2, template, threshold], name='processor')

    reader.start()
    processor.start()

    save_images(q2, out_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main('./data/video.avi', './data/image.png', './processed', threshold=0.99)
