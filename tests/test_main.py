import os
from pathlib import Path
from queue import Queue

import numpy as np
import pytest

import main

BASE_DIR = Path().absolute().parent


@pytest.fixture
def image():
    return np.zeros((16, 16))


@pytest.fixture
def big_image():
    return np.zeros((32, 32, 3))


@pytest.fixture
def real_image():
    return main.read_image(str(BASE_DIR / main.PATH_TO_IMAGE))


def test_read_image():
    image = main.read_image(str(BASE_DIR / main.PATH_TO_IMAGE))
    assert image.size != 0


def test_read_video():
    q = Queue()
    main.read_video(str(BASE_DIR / main.PATH_TO_VIDEO), q)
    assert not q.empty()


def test_save_image(image, tmp_path):
    file_name = str(tmp_path / 'test_image.png')
    main.save_image(image, file_name)
    assert os.path.exists(file_name)


def test_find_position(real_image):
    position = main.find_position(real_image, real_image, threshold=0.99)
    assert position == main.Vector(0, 0)


def test_draw_rectangle(big_image):
    big_image_old = big_image.copy()
    main.draw_rectangle(big_image, main.Vector(0, 0), main.Vector(5, 5))
    assert not np.array_equal(big_image_old, big_image)


def test_shape():
    assert main.shape(np.zeros((32, 16)) == main.Vector(16, 32))


def test_recreate_dir(tmp_path_factory):
    test_dir = tmp_path_factory.mktemp('test_dir')
    with open(test_dir / 'test_file', 'w') as f:
        f.write('tests_tests')
    main.recreate_dir(test_dir)
    assert len(os.listdir(test_dir)) == 0
