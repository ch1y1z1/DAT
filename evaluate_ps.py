from basicsr.utils import imwrite
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils.img_util import tensor2img
from basicsr.utils.options import parse_options
from tqdm import tqdm
import os.path as osp
import os
from metrics_psnr_ssim import evaluate as evaluate_psnr_ssim


def evaluate(root_path, dataset_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    opt, _ = parse_options(root_path, is_train=False)
    model = build_model(opt)

    dataset_opt = {
        "task": "SR",
        "name": "Urban100",
        "type": "PairedImageDataset",
        "dataroot_gt": dataset_path + "/HR",
        "dataroot_lq": dataset_path + "/LR",
        "filename_tmpl": "{}x4",
        "io_backend": {"type": "disk"},
        "phase": "val",
        "num_worker_per_gpu": 4,
        "batch_size_per_gpu": 1,
        "scale": 4,
        "manual_seed": 10,
    }
    dataset = build_dataset(dataset_opt)
    dataloader = build_dataloader(dataset, dataset_opt)

    for _idx, val_data in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Evaluating"
    ):
        img_name = osp.splitext(osp.basename(val_data["lq_path"][0]))[0]

        model.feed_data(val_data)
        model.test()
        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals["result"]])

        imwrite(sr_img, osp.join(output_path, f"{img_name}.png"))

    print(f"Evaluation complete. Results saved to {output_path}")

    evaluate_psnr_ssim(dataset_path + "/HR", output_path, 4)


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    evaluate(root_path, "../datasets/Urban100", "../results/Urban100")
