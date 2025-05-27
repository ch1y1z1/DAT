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
results_vol = modal.Volume.from_name("cf_training_results")

app = modal.App(name="mdlr")


@app.function(
    image=image, 
    gpu="A100-40GB", 
    volumes={"/datasets": vol, "/training_results": results_vol}, 
    timeout=3600
)
def train():
    from basicsr import train
    from os import path as osp
    import sys
    import shutil
    import datetime
    import os

    sys.argv = ["", "-opt", "/root/train_DAT_light_x4.yml"]

    # 执行训练
    train.train_pipeline(osp.abspath(osp.join(__file__, osp.pardir)))
    
    # 训练结束后，复制 experiments 文件夹到结果 volume
    experiments_path = "/root/experiments"
    if os.path.exists(experiments_path):
        # 生成时间戳
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = f"/training_results/experiments_{timestamp}"
        
        print(f"复制训练结果到: {target_path}")
        shutil.copytree(experiments_path, target_path)
        
        # 提交 volume 更改
        results_vol.commit()
        print("训练结果已成功保存到 volume")
    else:
        print("警告: experiments 文件夹不存在")


@app.local_entrypoint()
def main():
    train.remote()
