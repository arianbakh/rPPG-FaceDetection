import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import cv2
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shutil
import sys

from deepface import DeepFace
from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import modeling, preprocessing
from deepface.modules.verification import find_cosine_distance, find_threshold
from PIL import Image
from retinaface import RetinaFace
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform


mpl.rcParams['axes.linewidth'] = 0


def get_frames(video_path):
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


def create_data_structures(frames, video_file_name):
    db_path = os.path.join('dbs', video_file_name)
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    face_ds = {}
    frame_ds = []
    for frame_index, frame in enumerate(frames):
        detection_result = RetinaFace.detect_faces(frame)
        frame_faces = []
        for face_index, face_instance in enumerate(detection_result.values()):
            face_id = '%d_%d' % (frame_index, face_index)
            face_rect = face_instance['facial_area']
            face_ds[face_id] = {
                'face_rect': face_rect,
                'right_eye': face_instance['landmarks']['right_eye'],
                'left_eye': face_instance['landmarks']['left_eye'],
                'person_id': None,
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


def get_face_image(frame, face_rect, rotation_direction, rotation_angle, inflate=2):
    center_x, center_y, square_size = convert_coordinates(face_rect)

    # inflate bounding box
    square_size = round(square_size * inflate)

    # get inflated face image
    face_img = crop_image(frame, center_x, center_y, square_size)

    # rotate inflated face image
    rotated_face_img = Image.fromarray(face_img)
    rotated_face_img = np.array(
        rotated_face_img.rotate(
            rotation_direction * rotation_angle,
            resample=Image.BICUBIC,
            expand=True
        )
    )

    # detect one face in the new image 
    rotated_detection_result = RetinaFace.detect_faces(rotated_face_img)  # TODO Bug: this may be empty
    rotated_highest_score_face = max(
        rotated_detection_result.values(),  # TODO maybe include sort by larger area?
        key=lambda x: x['score']
    )  # only use one face with the highest score

    new_center_x, new_center_y, new_square_size = convert_coordinates(rotated_highest_score_face['facial_area'])

    # crop the face
    new_face_img = crop_image(rotated_face_img, new_center_x, new_center_y, new_square_size)

    return new_face_img


def add_face_images(face_ds, frame_ds, output_dir, video_file_name, sample_index):
    for i, (face_id, face_data) in enumerate(face_ds.items()):
        face_img = get_face_image(
            frame_ds[face_data['frame_index']]['frame_img'],
            face_data['face_rect'],
            face_data['rotation_direction'],
            face_data['rotation_angle']
        )
        face_ds[face_id]['face_img'] = face_img
        if i == sample_index:
            plt.imshow(face_img)
            plt.savefig(
                os.path.join(output_dir, '%s_face%d.png' % (video_file_name, sample_index))
            )


def add_person_embedding(face_ds):
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


def add_person_ids(face_ds, embeddings_index, output_dir, video_file_name):
    threshold = find_threshold('Facenet512', 'cosine') + 0.01
    distances = pdist(embeddings_index['embeddings'], metric=find_cosine_distance)
    distance_matrix = squareform(distances)
    adjacency_matrix = (distance_matrix < threshold).astype(int)
    sparse_matrix = csr_matrix(adjacency_matrix)
    num_components, labels = connected_components(csgraph=sparse_matrix, directed=False, connection='strong', return_labels=True)
    print('%d distinct people detected' % num_components)
    for i in range(len(labels)):
        face_id = embeddings_index['face_ids'][i]
        person_id = labels[i]
        face_ds[face_id]['person_id'] = person_id
    for i in range(num_components):
        component = np.where(labels == i)[0]
        rows = cols = int(math.ceil(math.sqrt(len(component))))
        fig, axes = plt.subplots(rows, cols, figsize=(rows * 5, cols * 5))
        for j in range(len(component)):
            row = j // rows
            col = j % rows
            node_index = component[j]
            node_face_id = embeddings_index['face_ids'][node_index]
            node_img = face_ds[node_face_id]['face_img']
            axes[row][col].imshow(node_img)
            axes[row][col].axis('off')
        for j in range(rows * cols - len(component), rows * cols):
            row = j // rows
            col = j % rows
            axes[row][col].axis('off')
        plt.savefig(os.path.join(output_dir, '%s_component%d' % (video_file_name, i)))


def get_clips(face_ds, frame_ds, off_screen_tolerance):
    clips = []
    active_person_ids = {}
    clip_counter = 0
    for frame_index, frame_data in enumerate(frame_ds):
        frame_face_ids = frame_data['face_ids']
        for face_id in frame_face_ids:
            person_id = face_ds[face_id]['person_id']
            if person_id not in active_person_ids:
                new_clip = {
                    'clip_id': clip_counter,
                    'start_frame': frame_index,
                    'person_id': person_id,
                    'face_ids': [{
                        'frame_index': frame_index,
                        'face_id': face_id,
                    }],
                }
                clip_counter += 1
                active_person_ids[person_id] = {
                    'off_screen': 0,
                    'touched_this_frame': True,
                    'clip': new_clip,
                }
            else:
                active_person_ids[person_id]['clip']['face_ids'].append({
                    'frame_index': frame_index,
                    'face_id': face_id,
                })
                active_person_ids[person_id]['touched_this_frame'] = True
        person_ids_to_remove = []
        for person_id in active_person_ids.keys():
            if not active_person_ids[person_id]['touched_this_frame']:
                active_person_ids[person_id]['off_screen'] += 1
            active_person_ids[person_id]['touched_this_frame'] = False
            if active_person_ids[person_id]['off_screen'] > off_screen_tolerance:
                person_ids_to_remove.append(person_id)
                clips.append(active_person_ids[person_id]['clip'])
        for person_id in person_ids_to_remove:
            del active_person_ids[person_id]
    for person_id in active_person_ids.keys():
        clips.append(active_person_ids[person_id]['clip'])
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
        for data in clip['face_ids']:
            face_id = data['face_id']
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
    filtered_clips = remove_short(clips, fps)
    filtered_clips = remove_spoofed(face_ds, clips)
    return filtered_clips


def get_clip_images(face_ds, clip):
    clip_imgs = []
    last_frame_index = -1
    max_size = -1
    for data in clip['face_ids']:
        frame_index = data['frame_index']
        face_id = data['face_id']
        if last_frame_index > 0:
            padding = frame_index - last_frame_index - 1
            last_img = clip_imgs[-1]
            clip_imgs += [last_img] * padding
        last_frame_index = frame_index
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
    clips_dir = get_clips_dir(output_dir, video_file_name)
    if os.path.exists(clips_dir):
        shutil.rmtree(clips_dir)
    os.makedirs(clips_dir)
    for clip in clips:
        clip_path = os.path.join(clips_dir, 'clip%d.avi' % (clip['clip_id']))
        clip_imgs = get_clip_images(face_ds, clip)
        save_face_video(clip_imgs, clip_path, fps)


def compress_clips(output_dir, video_file_name):
    clips_dir = get_clips_dir(output_dir, video_file_name)
    output_path = os.path.join(output_dir, '%s_clips' % video_file_name)
    shutil.make_archive(output_path, 'zip', clips_dir)
    shutil.rmtree(clips_dir)


def run(args):
    video_file_name = os.path.basename(args.video_path).split('.')[0]
    sample_index = 0
    fps, frames = get_frames(args.video_path)
    face_ds, frame_ds = create_data_structures(frames, video_file_name)
    add_align(face_ds)
    add_face_images(face_ds, frame_ds, args.output_dir, video_file_name, sample_index)
    embeddings_index = add_person_embedding(face_ds)
    add_person_ids(face_ds, embeddings_index, args.output_dir, video_file_name)
    off_screen_tolerance = fps // 10  # 0.1 seconds
    clips = get_clips(face_ds, frame_ds, off_screen_tolerance)
    filtered_clips = filter_clips(face_ds, clips, fps)
    render_clips(filtered_clips, face_ds, args.output_dir, video_file_name, fps)
    compress_clips(args.output_dir, video_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Path of input raw video')
    parser.add_argument('--output-dir', type=str, help='Path of output faces video')
    args = parser.parse_args()
    run(args)
