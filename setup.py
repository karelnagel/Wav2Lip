from setuptools import setup, find_packages

setup(
    name="wav2lip",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "librosa",
        "numpy",
        "opencv-contrib-python",
        "opencv-python",
        "torch",
        "torchvision",
        "tqdm",
        "numba",
    ],
)
