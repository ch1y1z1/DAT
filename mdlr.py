import modal

image = (
    modal.Image.debian_slim(python_version="3.9")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("basicsr")
    .add_local_file(
        "./train_DAT_light_x4.yml",
        remote_path="/root/train_DAT_light_x4.yml",
    )
)
vol = modal.Volume.from_name("cf_data")

app = modal.App(name="mdlr")


@app.function(image=image, gpu="A100-40GB", volumes={"/datasets": vol}, timeout=3600)
def train():
    from basicsr import train
    from os import path as osp
    import sys

    sys.argv = ["", "-opt", "/root/train_DAT_light_x4.yml"]

    train.train_pipeline(osp.abspath(osp.join(__file__, osp.pardir)))


@app.local_entrypoint()
def main():
    train.remote()
