from pathlib import Path
from typing import FrozenSet, Optional, Union

from nerfbaselines import UnloadedDataset, DatasetFeature
from nerfbaselines.datasets.colmap import load_colmap_dataset


def load_eth3d_dataset(
    path: Union[Path, str],
    split: str,
    features: Optional[FrozenSet[DatasetFeature]] = None,
    **kwargs,
):
    # Will use every 8th image from the train set as test frames by default
    dataset: UnloadedDataset = load_colmap_dataset(
        path,
        split,
        colmap_path="colmap",
        images_path="images",
        features=features,
        **kwargs,
    )

    dataset["metadata"]["dense_points3D_path"] = str(
        Path(path).absolute() / "scan_merged.ply"
    )
    return dataset


def download_eth3d_not_implemented():
    raise NotImplementedError(
        "This dataset loader does not support downloading datasets."
    )
