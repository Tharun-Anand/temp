import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

# Function to display a video in a Jupyter notebook
def display_video(tensor):
    """\
    Input: T * H * W * C
    Output: HTML5 video
    """
    # Ensure the tensor is in CPU and numpy format and permute it to T * H * W * C
    video = tensor.permute(1, 2, 3, 0).detach().cpu().numpy()  # T * H * W * C
    # Normalize the video if it is not in the expected range
    if video.dtype == np.float32:
        video = np.clip(video, 0, 1)  # Assuming the video is in [0, 1] range
    elif video.dtype == np.uint8:
        video = np.clip(video, 0, 255)  # Assuming the video is in [0, 255] range

  

    fig, ax = plt.subplots()
    im = ax.imshow(video[0])  # Initialize with the first frame
    ax.axis('off')

    def update(frame):
        im.set_data(video[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=video.shape[0], interval=50, blit=False)
    plt.close(fig)  # Prevents showing the static plot
    return ani.to_html5_video()