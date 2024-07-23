import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import argparse
import cv2
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shutil
import sys

from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import modeling, preprocessing
from deepface.modules.verification import find_cosine_distance, find_threshold
from multiprocessing import Process, Manager
from PIL import Image
from retinaface import RetinaFace
from retinaface.model import retinaface_model
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

import tensorflow as tf  # should be imported after retinaface


mpl.rcParams['axes.linewidth'] = 0


def get_frames(video_path):
    print('Loading video frames...')
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = video.read()
    frames = []
    while success:
        frame = np.asarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        success, frame = video.read()
    return fps, np.array(frames)


def face_detection_process(process_index, process_frames, processes_return_dict):
    model = tf.function(
        retinaface_model.build_model(),
        input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),),
    )
    detection_results = []
    for frame in process_frames:
        detection_result = RetinaFace.detect_faces(frame, model=model)
        detection_results.append(detection_result)
    processes_return_dict[process_index] = detection_results


def parallel_face_detection(images, num_processes):
    manager = Manager()
    processes_return_dict = manager.dict()
    process_image_count = int(math.ceil(len(images) / num_processes))
    processes = []
    for process_index in range(num_processes):
        process_images = images[
            process_index * process_image_count:(process_index + 1) * process_image_count
        ]
        process = Process(
            target=face_detection_process,
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
    detection_results = []
    sorted_processes_return_dict = sorted(processes_return_dict.items(), key=lambda x: int(x[0]))
    for process_index, process_detection_results in sorted_processes_return_dict:
        detection_results += process_detection_results
    return detection_results


def create_data_structures(frames, video_file_name, num_processes):
    print('Creating face and frame data structures...')
    db_path = os.path.join('dbs', video_file_name)
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    detection_results = parallel_face_detection(frames, num_processes)
    face_ds = {}
    frame_ds = []
    for frame_index, frame in enumerate(frames):
        detection_result = detection_results[frame_index]
        frame_faces = []
        for face_index, face_instance in enumerate(detection_result.values()):
            face_id = '%d_%d' % (frame_index, face_index)
            face_rect = face_instance['facial_area']
            face_ds[face_id] = {
                'face_rect': face_rect,
                'right_eye': face_instance['landmarks']['right_eye'],
                'left_eye': face_instance['landmarks']['left_eye'],
                'frame_index': frame_index,
            }
            frame_faces.append(face_id)
        frame_ds.append({
            'frame_img': frame,
            'face_ids': frame_faces,
        })
    return face_ds, frame_ds


def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_rotation(right_eye, left_eye):
    """
    from https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/
    """
    right_eye_x, right_eye_y = right_eye
    left_eye_x, left_eye_y = left_eye
    if left_eye_y < right_eye_y:
        point = (right_eye_x, left_eye_y)
        direction = -1 # clockwise
    else:
        point = (left_eye_x, right_eye_y)
        direction = 1  # counter-clockwise
    a = euclidean_distance(left_eye, point)
    b = euclidean_distance(right_eye, left_eye)
    c = euclidean_distance(right_eye, point)
    cos_a = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    angle = np.arccos(cos_a)  # radians
    angle = (angle * 180) / math.pi  # degrees
    if direction == -1:
        angle = 90 - angle
    return direction, angle


def add_align(face_ds):
    print('Aligning faces...')
    for face_id, face_data in face_ds.items():
        rotation_direction, rotation_angle = get_rotation(face_data['right_eye'], face_data['left_eye'])
        face_ds[face_id]['rotation_direction'] = rotation_direction
        face_ds[face_id]['rotation_angle'] = rotation_angle


def convert_coordinates(rect):
    """
    transforms (x0, y0, x1, y1) to (x, y, size, size)
    """
    x_min, y_min, x_max, y_max = rect
    x = x_min
    y = y_min
    width = x_max - x_min
    height = y_max - y_min
    center_x = x + width // 2
    center_y = y + height // 2
    square_size = max(width, height)
    return (center_x, center_y, square_size)


def bound(low, value, high):
    return max(low, min(high, value))


def crop_image(input_img, center_x, center_y, square_size):
    new_x = bound(0, center_x - (square_size // 2), input_img.shape[1] - square_size)
    new_y = bound(0, center_y - (square_size // 2), input_img.shape[0] - square_size)
    bounding_box = [new_x, new_y, square_size, square_size]
    output_img = input_img[
        bounding_box[1]:bounding_box[1] + bounding_box[3],
        bounding_box[0]:bounding_box[0] + bounding_box[2],
        :
    ]
    return output_img


def get_rotated_face_image(frame, face_rect, rotation_direction, rotation_angle, inflate=2):
    center_x, center_y, square_size = convert_coordinates(face_rect)

    # inflate bounding box
    inflated_square_size = round(square_size * inflate)

    # get inflated face image
    face_img = crop_image(frame, center_x, center_y, inflated_square_size)

    # rotate inflated face image
    rotated_face_img = Image.fromarray(face_img)
    rotated_face_img = np.array(
        rotated_face_img.rotate(
            rotation_direction * rotation_angle,
            resample=Image.BICUBIC,
            expand=True
        )
    )
    return rotated_face_img


def get_face_image(rotated_face_img, rotated_detection_result, frame, face_rect):
    if len(rotated_detection_result) > 0:
        rotated_highest_score_face = max(
            rotated_detection_result.values(),  # TODO maybe include sort by larger area?
            key=lambda x: x['score']
        )  # only use one face with the highest score

        new_center_x, new_center_y, new_square_size = convert_coordinates(rotated_highest_score_face['facial_area'])

        # crop the face
        new_face_img = crop_image(rotated_face_img, new_center_x, new_center_y, new_square_size)

        return new_face_img
    else:
        center_x, center_y, square_size = convert_coordinates(face_rect)
        original_face_img = crop_image(frame, center_x, center_y, square_size)
        return original_face_img


def add_face_images(face_ds, frame_ds, output_dir, video_file_name, sample_index, num_processes):
    print('Extracting face images...')
    rotated_face_imgs = []
    for i, (face_id, face_data) in enumerate(face_ds.items()):
        rotated_face_img = get_rotated_face_image(
            frame_ds[face_data['frame_index']]['frame_img'],
            face_data['face_rect'],
            face_data['rotation_direction'],
            face_data['rotation_angle']
        )
        rotated_face_imgs.append(rotated_face_img)
    rotated_detection_results = parallel_face_detection(rotated_face_imgs, num_processes)
    for i, (face_id, face_data) in enumerate(face_ds.items()):
        face_img = get_face_image(
            rotated_face_imgs[i],
            rotated_detection_results[i],
            frame_ds[face_data['frame_index']]['frame_img'],
            face_data['face_rect']
        )
        face_ds[face_id]['face_img'] = face_img
        if i == sample_index:
            plt.imshow(face_img)
            plt.savefig(
                os.path.join(output_dir, '%s_face%d.png' % (video_file_name, sample_index))
            )


def add_person_embedding(face_ds):
    print('Extracting person embeddings...')
    embeddings = []
    face_ids = []
    model: FacialRecognition = modeling.build_model('Facenet512')
    target_size = model.input_shape
    for i, (face_id, face_data) in enumerate(face_ds.items()): 
        face_img = face_data['face_img']
        processed_img = preprocessing.resize_image(img=face_img, target_size=(target_size[1], target_size[0]))
        processed_img = preprocessing.normalize_input(img=processed_img, normalization='base')
        embedding = model.forward(processed_img)
        face_ds[face_id]['embedding'] = embedding
        face_ids.append(face_id)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    embeddings_index = {
        'embeddings': embeddings,
        'face_ids': face_ids,
    }
    return embeddings_index


def get_embedding_distance(face_ds, face_id1, face_id2):
    embedding1 = face_ds[face_id1]['embedding']
    embedding2 = face_ds[face_id2]['embedding']
    return find_cosine_distance(embedding1, embedding2)


def get_iou_similarity(face_ds, face_id1, face_id2):
    x_min1, y_min1, x_max1, y_max1 = face_ds[face_id1]['face_rect']
    x_min2, y_min2, x_max2, y_max2 = face_ds[face_id2]['face_rect']
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height
    area_rect1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area_rect2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area_rect1 + area_rect2 - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def get_location_similarity(face_ds, frame_ds, face_id1, face_id2):
    face1_center_x = (face_ds[face_id1]['face_rect'][2] + face_ds[face_id1]['face_rect'][0]) / 2
    face1_center_y = (face_ds[face_id1]['face_rect'][3] + face_ds[face_id1]['face_rect'][1]) / 2
    face2_center_x = (face_ds[face_id2]['face_rect'][2] + face_ds[face_id2]['face_rect'][0]) / 2
    face2_center_y = (face_ds[face_id2]['face_rect'][3] + face_ds[face_id2]['face_rect'][1]) / 2
    center_distance = math.sqrt((face1_center_x - face2_center_x) ** 2 + (face1_center_y - face2_center_y) ** 2)
    frame1_index = face_ds[face_id1]['frame_index']
    frame1_img = frame_ds[frame1_index]['frame_img']
    frame1_width = frame1_img.shape[1]
    frame1_height = frame1_img.shape[0]
    max_distance = math.sqrt(frame1_width ** 2 + frame1_height ** 2)  # assuming all frame sizes are equal
    return 1 - center_distance / max_distance


def get_area_similarity(face_ds, face_id1, face_id2, epsilon=1e-8):
    face1_width = face_ds[face_id1]['face_rect'][2] - face_ds[face_id1]['face_rect'][0]
    face1_height = face_ds[face_id1]['face_rect'][3] - face_ds[face_id1]['face_rect'][1]
    face_1_area = face1_width * face1_height
    face2_width = face_ds[face_id2]['face_rect'][2] - face_ds[face_id2]['face_rect'][0]
    face2_height = face_ds[face_id2]['face_rect'][3] - face_ds[face_id2]['face_rect'][1]
    face_2_area = face2_width * face2_height
    return min(face_1_area / (face_2_area + epsilon), face_2_area / (face_1_area + epsilon))


def get_resolution_coefficient(face_ds, face_id1, face_id2):
    facenet512_input_size = 160
    face1_rect = face_ds[face_id1]['face_rect']
    face2_rect = face_ds[face_id2]['face_rect']
    face1_width = face1_rect[2] - face1_rect[0]
    face1_height = face1_rect[3] - face1_rect[1]
    face2_width = face2_rect[2] - face2_rect[0]
    face2_height = face2_rect[3] - face2_rect[1]
    resolution = min(face1_width, face1_height, face2_width, face2_height)
    return bound(0, resolution, facenet512_input_size) / facenet512_input_size


def get_face_similarity(face_ds, frame_ds, face_id1, face_id2):
    """
    The range all metrics is [0, 1]
    In all metrics, higher is better
    """
    embedding_distance = get_embedding_distance(face_ds, face_id1, face_id2)  # [0, 2] lower is better
    if embedding_distance <= find_threshold('Facenet512', 'cosine'):
        same_person = 1
    else:
        same_person = 0
    embedding_similarity = 1 - embedding_distance / 2  # [0, 1] higher is better
    resolution_coefficient = get_resolution_coefficient(face_ds, face_id1, face_id2)
    iou_similarity = get_iou_similarity(face_ds, face_id1, face_id2)
    location_similarity = get_location_similarity(face_ds, frame_ds, face_id1, face_id2)
    area_similarity = get_area_similarity(face_ds, face_id1, face_id2)
    deep_similarity = (same_person + embedding_similarity) * resolution_coefficient / 2
    classic_similarity = (iou_similarity + location_similarity * area_similarity) / 2
    face_similarity = (deep_similarity + classic_similarity) / 2 - 0.5  # TODO
    metrics = {
        'face_similarity': face_similarity,
        'embedding_similarity': embedding_similarity,
        'same_person': same_person,
        'resolution_coefficient': resolution_coefficient,
        'iou_similarity': iou_similarity,
        'location_similarity': location_similarity,
        'area_similarity': area_similarity,
        'deep_similarity': deep_similarity,
        'classic_similarity': classic_similarity,
    }
    return metrics


def get_clips(face_ds, frame_ds, output_dir, video_file_name):
    print('Generating clips...')
    clips_graph = nx.Graph()
    metric_lists = {
        'face_similarity': [],
        'embedding_similarity': [],
        'same_person': [],
        'resolution_coefficient': [],
        'iou_similarity': [],
        'location_similarity': [],
        'area_similarity': [],
        'deep_similarity': [],
        'classic_similarity': [],
    }
    for i in range(len(frame_ds) - 1):
        current_frame_face_ids = frame_ds[i]['face_ids']
        next_frame_face_ids = frame_ds[i + 1]['face_ids']
        if i == 0:
            clips_graph.add_nodes_from(current_frame_face_ids, frame=i)    
        clips_graph.add_nodes_from(next_frame_face_ids, frame=i + 1)
        for current_frame_face_id in current_frame_face_ids:
            for next_frame_face_id in next_frame_face_ids:
                metrics = get_face_similarity(face_ds, frame_ds, current_frame_face_id, next_frame_face_id)
                metric_lists['face_similarity'].append(metrics['face_similarity'])
                metric_lists['embedding_similarity'].append(metrics['embedding_similarity'])
                metric_lists['same_person'].append(metrics['same_person'])
                metric_lists['resolution_coefficient'].append(metrics['resolution_coefficient'])
                metric_lists['iou_similarity'].append(metrics['iou_similarity'])
                metric_lists['location_similarity'].append(metrics['location_similarity'])
                metric_lists['area_similarity'].append(metrics['area_similarity'])
                metric_lists['deep_similarity'].append(metrics['deep_similarity'])
                metric_lists['classic_similarity'].append(metrics['classic_similarity'])
                clips_graph.add_edge(current_frame_face_id, next_frame_face_id, weight=metrics['face_similarity'])
        nodes_subset = current_frame_face_ids + next_frame_face_ids
        subgraph = clips_graph.subgraph(nodes_subset).copy()
        max_matching = nx.max_weight_matching(subgraph, maxcardinality=False)
        edges_to_keep = set(max_matching)
        edges_to_keep_inverted = set([(a, b) for (b, a) in edges_to_keep])
        edges_to_remove = set(subgraph.edges()) - (edges_to_keep | edges_to_keep_inverted)  # TODO write with nx.complement
        clips_graph.remove_edges_from(edges_to_remove)
    for metric_name, metric_list in metric_lists.items():
        plt.close()
        plt.hist(metric_list, bins=20)
        plt.title(metric_name)
        plt.savefig(os.path.join(output_dir, '%s_hist_%s.png' % (video_file_name, metric_name)))
    plt.close()
    components = nx.connected_components(clips_graph)
    clips = []
    for clip_id, component in enumerate(components):
        nodes_with_frame = [
            (
                node_id,
                clips_graph.nodes[node_id]['frame']
            ) for node_id in component
        ]
        sorted_nodes = sorted(nodes_with_frame, key=lambda x: x[1])
        sorted_face_ids = [node_id for node_id, _ in sorted_nodes]
        clip = {
            'clip_id': clip_id,
            'face_ids': sorted_face_ids,
        }
        clips.append(clip)
    return clips


def remove_short(clips, length_threshold):
    filtered_clips = []
    for clip in clips:
        if len(clip['face_ids']) >= length_threshold:
            filtered_clips.append(clip)
    return filtered_clips


def remove_spoofed(face_ds, clips, vote_threshold=0.05):
    antispoof_model = modeling.build_model(model_name='Fasnet')
    filtered_clips = []
    for clip in clips:
        votes = []
        for face_id in clip['face_ids']:
            face_img = face_ds[face_id]['face_img']
            is_real, _ = antispoof_model.analyze(
                img=face_img,
                facial_area=(0, 0, face_img.shape[0], face_img.shape[1])
            )
            votes.append(is_real)
        mean_vote = np.mean(votes)
        if mean_vote >= vote_threshold:
            filtered_clips.append(clip)
    return filtered_clips


def filter_clips(face_ds, clips, fps):
    print('Filtering clips...')
    filtered_clips = remove_short(clips, fps)
    filtered_clips = remove_spoofed(face_ds, filtered_clips)
    return filtered_clips


def get_clip_images(face_ds, clip):
    clip_imgs = []
    max_size = -1
    for face_id in clip['face_ids']:
        face_img = face_ds[face_id]['face_img']
        clip_imgs.append(face_img)
        face_img_size = face_img.shape[0]  # assuming square
        if face_img_size > max_size:
            max_size = face_img_size
    resized_clip_imgs = []
    for clip_img in clip_imgs:
        resized_clip_img = cv2.resize(
            clip_img,
            (max_size, max_size),
            interpolation=cv2.INTER_CUBIC
        )
        resized_clip_imgs.append(resized_clip_img)
    return resized_clip_imgs


def save_face_video(clip_imgs, clip_path, fps):
    face_size = clip_imgs[0].shape[0]  # assuming all square and all equal
    video=cv2.VideoWriter(
        clip_path,
        0,
        fps,
        (face_size, face_size)
    )
    for face_img in clip_imgs:
        video.write(cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    video.release()


def get_clips_dir(output_dir, video_file_name):
    return os.path.join(output_dir, '%s_clips' % video_file_name)


def render_clips(clips, face_ds, output_dir, video_file_name, fps):
    print('Rendering clips...')
    clips_dir = get_clips_dir(output_dir, video_file_name)
    if os.path.exists(clips_dir):
        shutil.rmtree(clips_dir)
    os.makedirs(clips_dir)
    for clip in clips:
        clip_path = os.path.join(clips_dir, 'clip%d.avi' % (clip['clip_id']))
        clip_imgs = get_clip_images(face_ds, clip)
        save_face_video(clip_imgs, clip_path, fps)


def compress_clips(output_dir, video_file_name):
    print('Compressing clips...')
    clips_dir = get_clips_dir(output_dir, video_file_name)
    output_path = os.path.join(output_dir, '%s_clips' % video_file_name)
    shutil.make_archive(output_path, 'zip', clips_dir)
    shutil.rmtree(clips_dir)


def run(args):
    video_file_name = os.path.basename(args.video_path).split('.')[0]
    sample_index = 0
    fps, frames = get_frames(args.video_path)
    face_ds, frame_ds = create_data_structures(frames, video_file_name, args.num_processes)
    add_align(face_ds)
    add_face_images(face_ds, frame_ds, args.output_dir, video_file_name, sample_index, args.num_processes)
    embeddings_index = add_person_embedding(face_ds)
    clips = get_clips(face_ds, frame_ds, args.output_dir, video_file_name)
    filtered_clips = filter_clips(face_ds, clips, fps)
    render_clips(filtered_clips, face_ds, args.output_dir, video_file_name, fps)
    compress_clips(args.output_dir, video_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Path of input raw video')
    parser.add_argument('--output-dir', type=str, help='Path of output faces video')
    parser.add_argument('--num-processes', type=int, default=1, help='Number of parallel processes for DNNs (essentially batch size)')
    args = parser.parse_args()
    run(args)
