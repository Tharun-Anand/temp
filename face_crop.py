from ..imports import *
from .loading import read_yaml
from ..config import DEVICE, IMGSIZE

import pathlib
current_path = pathlib.Path(__file__).parent.resolve()

T = TypeVar("T")

class Singleton:
    all_instances: Dict[Type, object] = {}

    def __new__(cls, clazz: Type[T]) -> T:
        cls.all_instances[clazz] = clazz()
        return cls.all_instances[clazz]


def crop_with_padding(image: ndarray, x1: int, x2: int, y1: int, y2: int, pad_value: Union[int, float] = 0.,
    batch: bool = False
) -> ndarray:
    assert y2 > y1 and x2 > x1, "Should follow y2 > y1 and x2 > x1"

    if not batch:
        image = image[np.newaxis, ...]

    crop_shape = np.array([y2 - y1, x2 - x1])

    if len(image.shape) == 3:
        b, h, w = image.shape
        cropped = np.full((b, *crop_shape), pad_value, dtype=image.dtype)
    elif len(image.shape) == 4:
        b, h, w, c = image.shape
        cropped = np.full((b, *crop_shape, c), pad_value, dtype=image.dtype)
    else:
        raise ValueError("Invalid shape, the image should be one of following shapes: ([B,] H, W) or ([B,] H, W, C)")

    # compute cropped index of image
    image_y_start, image_x_start = np.clip([y1, x1], 0, [h, w])
    image_y_end, image_x_end = np.clip([y2, x2], 0, [h, w])

    # compute target index of output
    crop_y_start, crop_x_start = np.clip([-y1, -x1], 0, crop_shape)
    crop_y_end, crop_x_end = crop_shape - np.clip([y2 - h, x2 - w], 0, crop_shape)

    # assign values
    cropped[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end] = \
        image[:, image_y_start:image_y_end, image_x_start:image_x_end]

    return cropped if batch else cropped[0]



@Singleton
class FaceXZooFaceDetector:

    def __init__(self):
        self.faceDetModelHandler = None
        self.inited = False

    def init(self, face_sdk_path: Optional[str] = None, device: str = "cuda:0"):
        if face_sdk_path is not None:
            sys.path.append(face_sdk_path)
        else:
            face_sdk_path = os.path.join(current_path, "FaceXZoo")
            if os.path.exists(face_sdk_path):
                sys.path.append(face_sdk_path)
                print('FaceX-Zoo loaded')
        try:
            from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
            from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
        except ImportError:
            raise ImportError("FaceX-Zoo cannot be imported, please specify the path to the face_sdk path of FaceXZoo"
                              " or put it in the working directory.")

        model_conf = read_yaml(os.path.join(face_sdk_path, "config", "model_conf.yaml"))
        model_path = os.path.join(face_sdk_path, 'models')
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name = model_conf[scene][model_category]

        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
        model, cfg = faceDetModelLoader.load_model()
        self.faceDetModelHandler = FaceDetModelHandler(model, device, cfg)
        self.inited = True

    @staticmethod
    def install(path: Optional[str] = None) -> str:
        """
        Install FaceX-Zoo by clone from GitHub.

        Args:
            path (``str``, optional): The path to install FaceX-Zoo, default is "./FaceX-Zoo".

        Returns:
            ``str``: The path to the installed FaceX-Zoo.

        """
        path = path or "FaceX-Zoo"
        if os.path.exists(path):
            return path

        os.system(f"git clone --depth=1 https://github.com/ControlNet/FaceX-Zoo {path or ''}")
        return path

    def detect_face(self, image: ndarray):
        assert image.ndim == 3 and image.shape[2] == 3, "frame should be 3-dim"
        dets = self.faceDetModelHandler.inference_on_image(image)
        return dets

    def crop_face(self, frame, margin=1, x=0, y=0,imgsize=None) -> Tuple[ndarray, int, int, int]:
        dets = self.detect_face(frame)
        if len(dets) > 0:
            x1, y1, x2, y2, confidence = dets[0]
            # center
            x, y = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            margin = int(max(abs(x2 - x1), abs(y2 - y1)) / 2)
        # crop face
        face = crop_with_padding(frame, x - margin, x + margin, y - margin, y + margin, 0)
        face = cv2.resize(face, imgsize or IMGSIZE)
        return face, margin, x, y

    def crop_image(self, image_path: str, out_path: str, max_faces=1, margin=0) -> None:
        if max_faces > 1:
            raise NotImplementedError("Multiple faces are not supported yet.")

        frame = cv2.imread(image_path)
        dets = self.detect_face(frame)
        for det in dets[:max_faces]:
            x1, y1, x2, y2, _ = det
            cropped = crop_with_padding(frame, int(x1 - margin), int(x2 + margin), int(y1 - margin), int(y2 + margin))
            cv2.imwrite(out_path, cropped)

    def crop_video(self, video_path: str, out_path: str, frame_size: Optional[Union[int, Tuple[int, int]]] = None,
        margin=0, fourcc="mp4v"
    ) -> None:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise IOError("Cannot open video file: " + video_path)

        # infer frame size
        if frame_size is None:
            frame_size = self._infer_frame_size(video_path, margin)

        if type(frame_size) is int:
            frame_size = frame_size, frame_size

        fps = video.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, frame_size)
        x1, y1, x2, y2 = 0, 0, 1, 1
        while True:
            ret, frame = video.read()
            if not ret:
                break
            dets = self.detect_face(frame)
            if len(dets) > 0:
                x1, y1, x2, y2, confidence = dets[0]
                # center
                x, y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                side = int(max(abs(x2 - x1), abs(y2 - y1)))
                x1 = x - side // 2
                x2 = x + side // 2
                y1 = y - side // 2
                y2 = y + side // 2

            cropped = crop_with_padding(frame, int(x1 - margin), int(x2 + margin), int(y1 - margin), int(y2 + margin))
            resized = cv2.resize(cropped, frame_size)
            writer.write(resized)
        video.release()
        writer.release()

    def crop_image_dir(self, image_dir: str, out_dir: str, pattern="*.jpg", *args, **kwargs) -> None:
        all_images = glob.glob(os.path.join(image_dir, pattern), root_dir=image_dir)
        for image_path in all_images:
            out_path = os.path.join(out_dir, image_path)
            self.crop_image(image_path, out_path, *args, **kwargs)

    def crop_video_dir(self, video_dir: str, out_dir: str, pattern="*.mp4", *args, **kwargs) -> None:
        all_videos = glob.glob(os.path.join(video_dir, pattern), root_dir=video_dir)
        for video_path in all_videos:
            out_path = os.path.join(out_dir, video_path)
            self.crop_video(video_path, out_path, *args, **kwargs)

    def _infer_frame_size(self, video_path: str, margin: int = 0
    ) -> Tuple[int, int]:
        video = cv2.VideoCapture(video_path)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            dets = self.detect_face(frame)
            if len(dets) > 0:
                x1, y1, x2, y2, confidence = dets[0]
                # center
                side = int(max(abs(x2 - x1), abs(y2 - y1)))
                video.release()
                return side + 2 * margin, side + 2 * margin