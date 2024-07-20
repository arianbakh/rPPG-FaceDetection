import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

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


def get_faces(frames, larger_box_coefficient=1.2):
    faces = []
    max_square_size = 0
    face_frames = []
    for i, frame in enumerate(frames):
        sys.stdout.write('\rDetecting faces %d/%d' % (i + 1, len(frames)))
        sys.stdout.flush()
        detection_result = RetinaFace.detect_faces(frame)
        if len(detection_result) > 0:
            highest_score_face = max(detection_result.values(), key=lambda x: x['score'])
            x_min, y_min, x_max, y_max = highest_score_face['facial_area']
            x = x_min
            y = y_min
            width = x_max - x_min
            height = y_max - y_min
            center_x = x + width // 2
            center_y = y + height // 2
            square_size = round(max(width, height) * larger_box_coefficient)
            if square_size > max_square_size:
                max_square_size = square_size
            new_x = bound(0, center_x - (square_size // 2), frame.shape[1] - square_size)
            new_y = bound(0, center_y - (square_size // 2), frame.shape[0] - square_size)
            face_bounding_box = [new_x, new_y, square_size, square_size]
            face_frames.append(
                frame[
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

    faces, face_size = get_faces(frames)

    if args.save_samples:
        plt.imshow(faces[sample_frame_index])
        plt.savefig(
            os.path.join(args.output_dir, '%s_frame%d_face.png' % (video_file_name, sample_frame_index))
        )

    create_face_video(faces, args.output_dir, video_file_name, face_size, fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, help='input raw video path')
    parser.add_argument('--output-dir', type=str, help='output faces video directory')
    parser.add_argument('--save-samples', action='store_true', help='whether to save sample frames')
    args = parser.parse_args()
    run(args)
