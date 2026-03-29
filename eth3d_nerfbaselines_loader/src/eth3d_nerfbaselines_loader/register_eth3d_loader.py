from nerfbaselines import register

register(
    {
        "id": "eth3d",
        "load_dataset_function": "eth3d_nerfbaselines_loader.eth3d_loader:load_eth3d_dataset",
    }
)

register(
    {
        "id": "eth3d",
        "download_dataset_function": "eth3d_nerfbaselines_loader.eth3d_loader:download_eth3d_not_implemented",
        "evaluation_protocol": "default",
        "metadata": {},
    }
)
