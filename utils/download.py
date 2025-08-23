import os
import requests
from tqdm.auto import tqdm


# Pre-trained weights for YoloV7 model and demo video URL's/
WEIGHTS_URL = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt"   # ?dl=1"
VIDEO_URL = "https://github.com/RizwanMunawar/yolov7-object-tracking/releases/download/yolov7-object-tracking/demo.mp4"


def download(dest_path, url=None, file_name=None):
    """ Download model weights to a destination path from a given url. """
    url = url if url is not None else WEIGHTS_URL
    resp = requests.get(url, stream=True)

    os.makedirs(dest_path, exist_ok=True)
    if not file_name:
        file_name = os.path.basename(url)
    output = os.path.abspath(os.path.join(dest_path, file_name))

    total = int(resp.headers.get("content-length", 0))
    with open(output, "wb") as file, tqdm(
        desc=file_name,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def download_demo_video(url=VIDEO_URL, dest="assets"):
    """Download demo video for inference."""
    os.makedirs(dest, exist_ok=True)
    file_path = os.path.join(dest, os.path.basename(url))
    r = requests.get(url, stream=True)
    with open(file_path, "wb") as f, tqdm(total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(1024):
            f.write(chunk)
            bar.update(len(chunk))
    return file_path

