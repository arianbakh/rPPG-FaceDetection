import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from PIL import Image
from retinaface import RetinaFace


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


def bound(low, value, high):
    return max(low, min(high, value))


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


def get_faces(frames, inflate=1.2, align=True, sample_frame_index=None, sample_file_path=None):
    faces = []
    max_square_size = 0
    face_frames = []
    for i, frame in enumerate(frames):
        sys.stdout.write('\rDetecting faces %d/%d' % (i + 1, len(frames)))
        sys.stdout.flush()
        detection_result = RetinaFace.detect_faces(frame)
        if len(detection_result) > 0:
            highest_score_face = max(detection_result.values(), key=lambda x: x['score'])
            if align:
                rotation_direction, rotation_angle = get_rotation(
                    highest_score_face['landmarks']['right_eye'],
                    highest_score_face['landmarks']['left_eye']
                )
                rotated_frame = Image.fromarray(frame)
                rotated_frame = np.array(
                    rotated_frame.rotate(
                        rotation_direction * rotation_angle,
                        resample=Image.BICUBIC,
                        expand=True
                    )
                )
                if sample_frame_index == i:
                    plt.imshow(rotated_frame)
                    plt.savefig(sample_file_path)
                    plt.close()
                frame_to_use = rotated_frame
                rotated_detection_result = RetinaFace.detect_faces(rotated_frame)
                rotated_highest_score_face = max(rotated_detection_result.values(), key=lambda x: x['score'])
                highest_score_face_to_use = rotated_highest_score_face
            else:
                frame_to_use = frame
                highest_score_face_to_use = highest_score_face
            x_min, y_min, x_max, y_max = highest_score_face_to_use['facial_area']
            x = x_min
            y = y_min
            width = x_max - x_min
            height = y_max - y_min
            center_x = x + width // 2
            center_y = y + height // 2
            square_size = round(max(width, height) * inflate)
            if square_size > max_square_size:
                max_square_size = square_size
            new_x = bound(0, center_x - (square_size // 2), frame_to_use.shape[1] - square_size)
            new_y = bound(0, center_y - (square_size // 2), frame_to_use.shape[0] - square_size)
            face_bounding_box = [new_x, new_y, square_size, square_size]
            face_frames.append(
                frame_to_use[
                    face_bounding_box[1]:face_bounding_box[1] + face_bounding_box[3],
                    face_bounding_box[0]:face_bounding_box[0] + face_bounding_box[2],
                    :
                ]
            )
    print()  # newline

    resized_face_frames = []
    for i, face_frame in enumerate(face_frames):
        sys.stdout.write('\rResizing faces %d/%d' % (i + 1, len(face_frames)))
        sys.stdout.flush()
        resized_face_frame = cv2.resize(
            face_frame,
            (max_square_size, max_square_size),
            interpolation=cv2.INTER_CUBIC
        )
        resized_face_frames.append(resized_face_frame)
    print()  # newline
    return np.array(resized_face_frames), max_square_size


def create_face_video(faces, output_dir, video_file_name, face_size, fps):
    print('FPS:', fps)
    video=cv2.VideoWriter(
        os.path.join(output_dir, '%s_faces.avi' % video_file_name),
        0,
        fps,
        (face_size, face_size)
    )
    for face in faces:
        video.write(cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    video.release()


def run(args):
    fps, frames = get_frames(args.video_path)

    video_file_name = os.path.basename(args.video_path).split('.')[0]
    sample_frame_index = 0
    if args.save_samples:
        plt.imshow(frames[sample_frame_index])
        plt.savefig(
            os.path.join(args.output_dir, '%s_frame%d_raw.png' % (video_file_name, sample_frame_index))
        )

    faces, face_size = get_faces(
        frames,
        args.inflate,
        args.align,
        sample_frame_index,
        os.path.join(args.output_dir, '%s_frame%d_raw_rotated.png' % (video_file_name, sample_frame_index))
    )

    if args.save_samples:
        plt.imshow(faces[sample_frame_index])
        plt.savefig(
            os.path.join(args.output_dir, '%s_frame%d_face.png' % (video_file_name, sample_frame_index))
        )

    create_face_video(faces, args.output_dir, video_file_name, face_size, fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Path of input raw video')
    parser.add_argument('--output-dir', type=str, help='Path of output faces video')
    parser.add_argument('--save-samples', action='store_true', help='Whether to save sample frames')
    parser.add_argument('--align', action='store_true', help='Whether to align faces')
    parser.add_argument('--inflate', type=float, default=1.2, help='How much to inflate bounding box')
    args = parser.parse_args()
    run(args)
