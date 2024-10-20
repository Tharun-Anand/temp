import cv2
import dlib
import torch
import numpy as np

# Load the dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/sushanth/deepfake_detection/tharun/shape_predictor_68_face_landmarks.dat")  # You need to download this model file

class SelfBlender:
    def __init__(self):
        # Initialize any transforms and other elements here
        pass

    def randaffine(self, img, mask):
        # Implement the randaffine method from SBI_Dataset
        pass

    def self_blending(self, img, landmark):
        H, W = len(img), len(img[0])
        mask = np.zeros_like(img[:, :, 0])
        cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        source = img.copy()
        if np.random.rand() < 0.5:
            source = self.transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source, mask)
        img_blended = source * mask[:, :, np.newaxis] + img * (1 - mask[:, :, np.newaxis])

        return img, img_blended, mask


def get_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None  # No face detected

    # Assume the first face detected is the one we want
    face = faces[0]
    landmarks = predictor(gray, face)

    # Convert the landmarks to a numpy array
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    return coords


def read_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"Error opening video file {path}")
        return None, None

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * 4)  # Number of frames in 4 seconds

    frame_count = 0
    self_blender = SelfBlender()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        
        landmarks = get_landmarks(frame)
        if landmarks is not None:
            # Perform self-blending
            _, frame_blended, _ = self_blender.self_blending(frame, landmarks)
            frames.append(frame_blended)
        else:
            frames.append(frame)  # If no landmarks are found, use the original frame

        frame_count += 1

    cap.release()

    if len(frames) == 0:
        print(f"No frames read from video file {path}")
        return None, None

    video = np.stack(frames)
    c = torch.from_numpy(video).permute(3, 0, 1, 2)  # C * T * H * W

    audio = torch.zeros(2, 1000)  # dummy
    return c, audio, {"video_fps": fps, "audio_fps": 10}

# Example usage
path = "your_video_path.mp4"
video_tensor, audio_tensor, meta = read_video(path)
