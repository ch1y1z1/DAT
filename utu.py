import modal

vol = modal.Volume.from_name("cf_data")

app = modal.App(name="utu")


@app.function(volumes={"/datasets": vol})
def prepare_data():
    import os
    import shutil

    os.makedirs("/datasets/DIV2K", exist_ok=True)
    # 移动所有文件到DIV2K文件夹
    for item in os.listdir("/datasets"):
        if item != "DIV2K":  # 跳过目标文件夹本身
            src_path = os.path.join("/datasets", item)
            dst_path = os.path.join("/datasets/DIV2K", item)
            shutil.move(src_path, dst_path)


@app.local_entrypoint()
def main():
    prepare_data.remote()
