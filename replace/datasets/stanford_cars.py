import pathlib
from typing import Callable, Optional, Any, Tuple

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class StanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        prompt_template = "A photo of a {}."
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        self._annotations_mat_path = self._base_folder / "cars_annos.mat"
        self._images_base_path = self._base_folder
        if self._split == "train":
            self._samples = [
                (
                    str(self._images_base_path / annotation["relative_im_path"]),
                    annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
                )
                for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
                if annotation["test"] == 0
            ]
        else:
            self._samples = [
                (
                    str(self._images_base_path / annotation["relative_im_path"]),
                    annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
                )
                for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
                if annotation["test"] == 1
            ]            


        self.classes = sio.loadmat(self._annotations_mat_path, squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.prompt_template = prompt_template
        self.clip_prompts = [ 
            prompt_template.format(label[:-5].lower().replace('_', ' ').replace('-', ' ')) \
            for label in self.classes
        ]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target