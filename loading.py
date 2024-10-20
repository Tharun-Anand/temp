from ..imports import *


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as file:
        return yaml.load(file, Loader=yaml.Loader)
    
import numpy as np
import torch


def add_gaussian_noise_psnr(video: torch.Tensor, target_psnr: float) -> torch.Tensor:
    """
    Adds Gaussian noise to the video frames to achieve a target PSNR.

    Parameters:
    - video: torch.Tensor of shape (C, T, H, W)
    - target_psnr: Desired PSNR (in dB)
    
    Returns:
    - noisy_video: torch.Tensor with added Gaussian noise
    """
    # Find the maximum pixel value in the video
    peak_signal_value = video.max()

    # Calculate mean squared error (MSE) for the target PSNR
    mse_target = (peak_signal_value ** 2) / (10 ** (target_psnr / 10))

    # Standard deviation of the Gaussian noise
    noise_std = torch.sqrt(mse_target)

    # Generate Gaussian noise with calculated standard deviation
    noise = torch.normal(0, noise_std, size=video.shape)

    # Add noise to the video
    noisy_video = video + noise

    return noisy_video

    

# def read_video(path: str):
#     """\
#     Visual:    C * T * H * W
#     Audio:     C * T | C = 1 if mono, 2 if stereo
#     info:      {'video_fps': VFPS, 'audio_fps': AFPS}
#     """
#     video, audio, info = torchvision.io.read_video(path)
#     video = rearrange(video, 'T H W C -> C T H W')
#     return video, audio, info



# def read_video(path):
#     cap = cv2.VideoCapture(path)

#     if not cap.isOpened():
#         print(f"Error opening video file {path}")
#         return None, None

#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
#         frames.append(frame)

#     cap.release()

#     video = np.stack(frames)
#     video = torch.from_numpy(video).permute(3,0,1,2)  # C * T * H * W

#     audio = torch.zeros(2,1000) # dummy
#     # print(video.shape)
#     return video, audio, {"video_fps": cap.get(cv2.CAP_PROP_FPS), "audio_fps": 10}

def read_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print(f"Error opening video file {path}")
        return None, None

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * 4)  # Number of frames in 4 seconds

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        frames.append(frame)
        frame_count += 1

    cap.release()

    if len(frames) == 0:
        print(f"No frames read from video file {path}")
        return None, None

    video = np.stack(frames)
    c = torch.from_numpy(video).permute(3, 0, 1, 2)  # C * T * H * W
    #c = add_gaussian_noise_psnr(c, target_psnr=300)

    audio = torch.zeros(2, 1000)  # dummy
    return c, audio, {"video_fps": fps, "audio_fps": 10}