import os
from torch import Tensor
from typing import Optional, Union
from torch.utils.data import Dataset
from torchvision.tv_tensors import Mask
from torchvision.io.image import decode_image
from torchvision.transforms.v2 import Transform


class PMDDataset(Dataset):
    def __init__(
        self, data_dir: Union[str, os.PathLike], transform: Optional[Transform] = None
    ) -> None:
        self.data_dir = data_dir
        self.transform = transform

        self.images = []
        for root, _, files in os.walk(self.data_dir):
            # Only list images if its mask exist
            if "image" in root and len(files) > 0:
                self.images.extend(
                    [
                        os.path.join(root, f)
                        for f in files
                        if f.endswith(".jpg")
                        and os.path.isfile(
                            os.path.join(
                                root.replace("image", "mask"), f.replace(".jpg", ".png")
                            )
                        )
                    ]
                )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path = self.images[idx]
        mask_path = img_path.replace(".jpg", ".png").replace("image", "mask")

        image = decode_image(img_path, "RGB").float() / 255.0
        mask = Mask((decode_image(mask_path) > 0).float())

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
