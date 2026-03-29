# ETH3D Dataset Integration for `ivd_splat`
This repository provides ETH3D (MVS) dataset integration for the `ivd_splat` implementation for the paper 
"The Role and Relationship of Initialization and Densification in 3D Gaussian Splatting" (https://arxiv.org/abs/2603.20714).
It contains a script that downloads and prepares the dataset and provides a loader for NerfBaselines
which allows to use it with our `ivd_splat` 3DGS implementation.
<!-- TODO: add main repo URL -->

# Usage
0. Install Python requirements from `requirements.txt`
0. Run `python prepare_dataset.py`. The script will:
    1. Download the required scene data - laser scans, DSLR images, for the scenes we used in our paper.
    2. Unzip the images and reduce their resolution by a factor of 4.
    2. Preprocess the laser scan data by merging separate scans into single .ply files
    3. Reorganize the files in a way that is expected by the nerfbaselines loader.
    - **NOTE:** The script overwrites the data in-place. While there is some handling for skipping already downloaded data, it is not robust, so in case of e.g. network errors, it may be necessary to delete the eth3d_dataset directory (or specific scene directories) and restart the process.
0. Once the script finishes, the data is ready, to use it with `ivd_splat`, do the following:
    1. Set `ETH3D_PATH` environment variable to the absolute path to the `eth3d_dataset` directory.
    2. Install `eth3d_nerfbaselines_loader` in the same python environment where `ivd_splat` will be invoked.
    3. Register `eth3d_nerfbaselines_loader` with nerfbaselines, e.g. by addding it to `NERFBASELINES_REGISTER` env. variable, e.g.:
        ```bash
        export NERFBASELINES_REGISTER="<THIS_REPO>/eth3d_nerfbaselines_loader/src/eth3d_nerfbaselines_loader/register_scannetpp_loader.py:$NERFBASELINES_REGISTER"
        ```
        Alternatively, see the `.envrc` file in the main repository that does this automatically,
        as long as `eth3d_nerfbaselines_loader` is installed.

Now, we can train with this dataset or generate initialization data from images by passing `eth3d` as the dataset id to `ivd_splat_runner` or `init_runner` respectively. 
```shell
init_runner --datasets eth3d --method monodepth --output-dir $RESULTS_DIR
```
Specific scenes can also be specified like this:
```shell
ivd_splat_runner --scenes eth3d/terrace eth3d/pipes \
        --output-dir $RESULTS_DIR \
        --configs "strategy={DefaultWithoutADCStrategy}" \
        --init_methods sfm
```

See main documentation for detailed instructions.


        






