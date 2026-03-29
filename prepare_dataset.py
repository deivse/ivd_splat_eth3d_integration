from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from PIL import Image  # type: ignore
from typing import Optional

import numpy as np
import open3d  # type: ignore
import py7zr
import py7zr.callbacks
import requests
import tyro
from tqdm import tqdm
from xml.etree import ElementTree as XMLElementTree

from nerfbaselines.datasets import _colmap_utils as colmap_utils

SCENES = [
    "courtyard",
    "delivery_area",
    "electro",
    "facade",
    "kicker",
    "meadow",
    "office",
    "pipes",
    "playground",
    "relief",
    "relief_2",
    "terrace",
    "terrains",
]


def load_pointcloud_ply(
    path: Path | str,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Loads a point cloud from a PLY file.

    Returns:
        pts: Nx3 array of point coordinates
        rgbs: Nx3 array of point colors (float, <0, 1> range), or None if no colors are present
    """
    pcd = open3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    rgbs = None
    if pcd.has_colors():
        rgbs = np.asarray(pcd.colors)
    if pts is None:
        raise RuntimeError(
            f"Failed to load point cloud from {path}, see open3d log messages."
        )
    return pts, rgbs


def export_pointcloud_ply(
    pts: np.ndarray,
    rgbs: Optional[np.ndarray],
    path: Path | str,
):
    """Exports a point cloud to a PLY file."""
    pcd = open3d.geometry.PointCloud()

    pcd.points = open3d.utility.Vector3dVector(pts)

    if rgbs is not None:
        if rgbs.max() > 1.0:
            rgbs = rgbs.astype(np.float64) / 255.0
        pcd.colors = open3d.utility.Vector3dVector(rgbs)

    open3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def process_scan(input_dir: Path, output_file: Path):
    """
    Loads transformations for each point cloud in input_dir from "scan_alignment.mlp", applies them to the point clouds, and
    saves a single point cloud to `output_file`.
    """

    # Load transformation matrices from scan_alignment.mlp
    mlp_file = input_dir / "scan_alignment.mlp"
    tree = XMLElementTree.parse(mlp_file)
    root = tree.getroot()

    # Collect all point clouds with their transformations
    all_pts = []
    all_rgbs = []

    for mesh in root.findall(".//MLMesh"):
        ply_filename = mesh.get("filename")
        ply_path = input_dir / ply_filename

        # Parse transformation matrix
        matrix_str = mesh.find(".//MLMatrix44").text
        matrix = np.array(
            [list(map(float, line.split())) for line in matrix_str.strip().split("\n")]
        )

        # Load point cloud
        pts, rgbs = load_pointcloud_ply(ply_path)

        # Apply transformation: add homogeneous coordinate, multiply by matrix, extract xyz
        pts_homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_transformed = (matrix @ pts_homogeneous.T).T[:, :3]

        all_pts.append(pts_transformed)
        if rgbs is not None:
            all_rgbs.append(rgbs)

    # Merge all point clouds
    merged_pts = np.vstack(all_pts)
    merged_rgbs = np.vstack(all_rgbs) if all_rgbs else None

    # Export merged point cloud
    export_pointcloud_ply(merged_pts, merged_rgbs, output_file)


@dataclass
class Args:
    output_dir: Path = Path("eth3d_dataset")
    max_workers_scans: int = -1
    undistorted_jpg_url: str = (
        "https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z"
    )
    scans_url: str = "https://www.eth3d.net/data/multi_view_training_dslr_scan_eval.7z"


# This is totally broken but whatever
class Py7ZrExtractProgressCallback(py7zr.callbacks.ExtractCallback):
    def __init__(self, description: str):
        self.tqdm_instance = tqdm(total=0, unit_scale=True, desc=description)
        self.total_decompressed_bytes = 0

    def report_start(self, processing_file_path, processing_bytes: str):
        pass

    def report_update(self, decompressed_bytes: str) -> None:
        self.total_decompressed_bytes += int(decompressed_bytes)
        self.tqdm_instance.update(self.total_decompressed_bytes)

    def report_start_preparation(self) -> None:
        pass

    def report_end(self, processing_file_path: str, wrote_bytes: str) -> None:
        pass

    def report_warning(self, message: str) -> None:
        pass

    def report_postprocess(self) -> None:
        pass


def download_and_extract(url: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    archive_path = output_dir / filename
    flag_path = output_dir / f".downloaded_{filename}"

    if flag_path.exists() and flag_path.read_text() == url:
        print(f"Already downloaded {url}, skipping.")
        return

    print(f"Downloading {url} to {archive_path}...")
    with open(archive_path, "wb") as f:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Extracting {archive_path} to {output_dir}...")

    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(
            path=output_dir,
            callback=Py7ZrExtractProgressCallback(f"Extracting {archive_path.name}"),
        )

    archive_path.unlink()
    with flag_path.open("w") as f:
        f.write(url)


def rename_stuff(base_dir: Path):
    def should_rename_colmap(old_path: Path, new_path: Path) -> bool:
        if not old_path.exists() and not new_path.exists():
            logging.warning(
                f"Missing path {old_path.name} for scene {old_path.parent.name}, skipping rename."
            )
            return False
        elif new_path.exists():
            logging.info(
                f"Path {new_path.name} already exists for scene {new_path.parent.name}, skipping rename."
            )
            return False
        return True

    def should_rename_images(old_path: Path, new_path: Path) -> bool:
        if not old_path.exists() and not new_path.exists():
            logging.warning(
                f"Missing path {old_path.name} for scene {old_path.parent.name}, skipping rename."
            )
            return False
        if not old_path.exists() and new_path.exists():
            logging.info(
                f"Path {new_path.name} already exists for scene {new_path.parent.name}, skipping rename."
            )
            return False
        return True

    for scene_dir in [d for d in base_dir.iterdir() if d.is_dir()]:
        colmap_old, colmap_new = (
            scene_dir / "dslr_calibration_undistorted",
            scene_dir / "colmap",
        )
        images_old, images_tmp = (
            scene_dir / "images" / "dslr_images_undistorted",
            scene_dir / "images_tmp",
        )
        images_new = scene_dir / "images"

        if should_rename_colmap(colmap_old, colmap_new):
            colmap_old.rename(colmap_new)

        # Different function since "images" dir already exits at the start
        if should_rename_images(images_old, images_new):
            images_old.rename(images_tmp)
            images_tmp.rename(images_new)

        # in colmap/images.txt, remove "dslr_images_undistorted/" prefix from image paths
        if colmap_new.exists():
            images_txt_path = colmap_new / "images.txt"
            print(
                f"Fixing image paths in {images_txt_path} for scene {scene_dir.name}..."
            )
            if not images_txt_path.exists():
                logging.warning(
                    f"Missing images.txt in {colmap_new} for scene {scene_dir.name}, cannot fix image paths in colmap data."
                )
            else:
                with images_txt_path.open("r") as f:
                    lines = f.readlines()
                with images_txt_path.open("w") as f:
                    for line in lines:
                        if line.startswith("#") or line.strip() == "":
                            f.write(line)
                            continue
                        f.write(line.replace("dslr_images_undistorted/", ""))


def write_nb_infos(base_dir: Path):
    for scene_dir in [d for d in base_dir.iterdir() if d.is_dir()]:
        nb_infos_path = scene_dir / "nb-info.json"

        content = {
            "loader": "eth3d",
            "id": "eth3d",
            "scene": scene_dir.name,
        }
        with nb_infos_path.open("w") as f:
            json.dump(content, f, indent=4)


def div_round_half_up(x, a):
    q, r = divmod(x, a)
    if 2 * r >= a:
        q += 1
    return q


def downscale_cameras(
    cameras_path: Path, output_cameras_path: Path, downscale_factor: int
):

    cameras = (
        colmap_utils.read_cameras_text(cameras_path)
        if cameras_path.name.endswith(".txt")
        else colmap_utils.read_cameras_binary(cameras_path)
    )
    new_cameras = {}
    for k, v in cameras.items():
        assert v.model == "PINHOLE", f"Expected PINHOLE camera model, got {v.model}."
        params = v.params
        oldw, oldh = v.width, v.height
        w = div_round_half_up(v.width, downscale_factor)
        h = div_round_half_up(v.height, downscale_factor)
        multx, multy = np.array([w, h], dtype=np.float64) / np.array(
            [oldw, oldh], dtype=np.float64
        )
        multipliers = np.stack([multx, multy, multx, multy], -1)
        params = params * multipliers
        new_camera = colmap_utils.Camera(
            id=v.id,
            model=v.model,
            width=w,
            height=h,
            params=params,
        )
        new_cameras[k] = new_camera

    # Write output
    os.makedirs(os.path.dirname(output_cameras_path), exist_ok=True)
    if output_cameras_path.suffix == ".txt":
        colmap_utils.write_cameras_text(new_cameras, output_cameras_path)
    else:
        colmap_utils.write_cameras_binary(new_cameras, output_cameras_path)


def downsample_images_and_adjust_colmap_for_downsampled_images(
    data_dir: Path, factor: int
):
    print("Downsampling images...")
    scenes = [d for d in data_dir.iterdir() if d.is_dir()]

    for scene_dir in tqdm(scenes, desc="Downsampling"):
        images_dir = scene_dir / "images"
        colmap_dir = scene_dir / "colmap"

        if not images_dir.exists() or not colmap_dir.exists():
            logging.warning(
                f"Missing images or colmap dir for scene {scene_dir.name}, skipping downsampling."
            )
            continue

        # Downsample images
        for img_path in images_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
            ]:
                with Image.open(img_path) as img:
                    new_size = (
                        div_round_half_up(img.width, factor),
                        div_round_half_up(img.height, factor),
                    )
                    img_resized = img.resize(new_size, resample=Image.LANCZOS)
                    img_resized.save(img_path, format="JPEG", quality=95, subsampling=0)

        downscale_cameras(
            scene_dir / "colmap" / "cameras.txt",
            scene_dir / "colmap" / "cameras.txt",
            downscale_factor=factor,
        )


def main():
    args = tyro.cli(Args)

    with ProcessPoolExecutor() as executor:
        download_and_extract_images = executor.submit(
            download_and_extract, args.undistorted_jpg_url, args.output_dir
        )

        download_extract_scans_futures = []
        for scene in SCENES:
            future = executor.submit(
                download_and_extract,
                f"https://www.eth3d.net/data/{scene}_scan_clean.7z",
                args.output_dir,
            )
            download_extract_scans_futures.append(future)

    for future in download_extract_scans_futures:
        future.result()  # Wait for scans to be downloaded and extracted before processing
    print("Finished downloading and extracting scans. Processing scans...")

    max_memory_to_use = 8  # GB
    memory_per_scan = 3  # GB per scan (this is a very rough, hopefully conservative estimate, adjust as needed)
    max_workers_by_memory = max(
        1, min(os.cpu_count() or 2, max_memory_to_use // memory_per_scan)
    )

    max_workers = (
        args.max_workers_scans if args.max_workers_scans > 0 else max_workers_by_memory
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        scan_processing_futures: list[Future] = []
        for scene_dir in [d for d in args.output_dir.iterdir() if d.is_dir()]:
            scans_dir = scene_dir / "scan_clean"
            output_file = scene_dir / "scan_merged.ply"
            if output_file.exists():
                print(
                    f"Output file {output_file} already exists, skipping scan processing for scene {scene_dir.name}."
                )
                continue

            scan_processing_futures.append(
                executor.submit(process_scan, scans_dir, output_file)
            )

        with tqdm(total=len(scan_processing_futures)) as pbar:
            for future in scan_processing_futures:
                future.result()  # Wait for all scan processing to finish
                pbar.update(1)

    download_and_extract_images.result()  # Wait for images to be downloaded and extracted before exiting

    rename_stuff(args.output_dir)
    write_nb_infos(args.output_dir)
    downsample_images_and_adjust_colmap_for_downsampled_images(
        args.output_dir, factor=4
    )
    print("All done!")


if __name__ == "__main__":
    main()
