import math
import numpy as np
import os
import sys
import traceback

from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import modeling, preprocessing
from multiprocessing import Process, Manager
from retinaface import RetinaFace
from retinaface.model import retinaface_model

import tensorflow as tf  # should be imported after retinaface


def face_detection_process(process_index, process_images, processes_return_dict):
    model = tf.function(
        retinaface_model.build_model(),
        input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),),
    )
    detection_results = []
    for image in process_images:
        detection_result = RetinaFace.detect_faces(image, model=model)
        detection_results.append(detection_result)
    processes_return_dict[process_index] = detection_results


def embedding_process(process_index, process_images, processes_return_dict):
    model: FacialRecognition = modeling.build_model('Facenet512')
    target_size = model.input_shape
    embeddings = []
    for image in process_images:
        processed_img = preprocessing.resize_image(img=image, target_size=(target_size[1], target_size[0]))
        processed_img = preprocessing.normalize_input(img=processed_img, normalization='base')
        embedding = model.forward(processed_img)
        embeddings.append(embedding)
    processes_return_dict[process_index] = embeddings


def antispoof_process(process_index, process_images, processes_return_dict):
    antispoof_model = modeling.build_model(model_name='Fasnet')
    is_reals = []
    for image in process_images:
        is_real, _ = antispoof_model.analyze(
            img=image,
            facial_area=(0, 0, image.shape[0], image.shape[1])
        )
        is_reals.append(is_real)
    processes_return_dict[process_index] = is_reals


def parallel_inference(images, num_processes, process_function):
    manager = Manager()
    processes_return_dict = manager.dict()
    process_image_count = int(math.ceil(len(images) / num_processes))
    processes = []
    for process_index in range(num_processes):
        process_images = images[
            process_index * process_image_count:(process_index + 1) * process_image_count
        ]
        process = Process(
            target=process_function,
            args=(process_index, process_images, processes_return_dict)
        )
        processes.append(process)
        process.start()
    exit_codes = []
    for process in processes:
        process.join()
        exit_codes.append(process.exitcode)
    for process_index, exit_code in enumerate(exit_codes):
        if exit_code != 0:
            raise Exception('Error: process %d has exit code %d' % (process_index, exit_code))
    results = []
    sorted_processes_return_dict = sorted(processes_return_dict.items(), key=lambda x: int(x[0]))
    for process_index, process_results in sorted_processes_return_dict:
        results += process_results
    return results
