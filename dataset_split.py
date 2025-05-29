import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分割配对图像数据集 (例如用于超分辨率).")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="包含HR和LR子文件夹的输入数据集根目录.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="保存分割后数据集的输出目录.")
    parser.add_argument("--hr_subdir", type=str, default="HR",
                        help="HR图像子文件夹的名称 (默认: 'HR').")
    parser.add_argument("--lr_subdir", type=str, default="LR",
                        help="LR图像子文件夹的名称 (默认: 'LR').")
    parser.add_argument("--lr_suffix", type=str, default="x4",
                        help="LR图像文件名的后缀 (例如, HR:'0001.png', LR:'0001x4.png' 则后缀为'x4'. 如果LR与HR文件名相同,则留空).")
    parser.add_argument("--img_ext", type=str, default=".png",
                        help="图像文件的扩展名 (默认: '.png').")
    parser.add_argument("--split_ratio", type=str, required=True,
                        help="逗号分隔的分割比例, 例如 '0.8,0.1,0.1' 用于 train,val,test.")
    parser.add_argument("--split_names", type=str, default="train,val,test",
                        help="逗号分隔的分割名称, 例如 'train,val,test'. 必须与split_ratio中的比例数量相匹配.")
    parser.add_argument("--seed", type=int, default=42,
                        help="用于复现的随机种子 (默认: 42).")
    return parser.parse_args()

def get_image_pairs(hr_dir: Path, lr_dir: Path, lr_suffix: str, img_ext: str) -> List[Tuple[Path, Path]]:
    """
    查找HR和LR图像对.
    HR图像名: <basename><img_ext>
    LR图像名: <basename><lr_suffix><img_ext>
    """
    pairs = []
    hr_images = sorted(list(hr_dir.glob(f"*{img_ext}")))
    
    if not hr_images:
        print(f"警告: 在 {hr_dir} 中未找到扩展名为 {img_ext} 的HR图像.")
        return []

    print(f"在 {hr_dir} 中找到 {len(hr_images)} 个HR图像.")

    for hr_path in hr_images:
        basename = hr_path.stem
        if lr_suffix:
            lr_filename = f"{basename}{lr_suffix}{img_ext}"
        else:
            lr_filename = f"{basename}{img_ext}"
        
        lr_path = lr_dir / lr_filename
        
        if lr_path.exists():
            pairs.append((hr_path, lr_path))
        else:
            print(f"警告: 找不到HR图像 {hr_path.name} 对应的LR图像 {lr_path.name}")
            
    print(f"成功匹配 {len(pairs)} 对HR/LR图像.")
    return pairs

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    hr_subdir_name = args.hr_subdir
    lr_subdir_name = args.lr_subdir
    lr_suffix = args.lr_suffix
    img_ext = args.img_ext
    seed = args.seed

    # 解析分割比例和名称
    try:
        split_ratios_str = args.split_ratio.split(',')
        split_ratios = [float(r) for r in split_ratios_str]
        if not (0.999 < sum(split_ratios) < 1.001): # 允许小的浮点误差
             raise ValueError(f"分割比例之和 ({sum(split_ratios)}) 不接近1.0. 请检查 --split_ratio.")
    except ValueError as e:
        print(f"错误: 无效的split_ratio参数: {args.split_ratio}. {e}")
        return

    split_names_str = args.split_names.split(',')
    if len(split_ratios) != len(split_names_str):
        print(f"错误: split_ratio的数量 ({len(split_ratios)}) 与split_names的数量 ({len(split_names_str)}) 不匹配.")
        return
    
    split_config = list(zip(split_names_str, split_ratios))

    # 设置随机种子
    random.seed(seed)
    print(f"使用随机种子: {seed}")

    # 构建HR和LR目录路径
    hr_dir_path = input_dir / hr_subdir_name
    lr_dir_path = input_dir / lr_subdir_name

    if not hr_dir_path.is_dir():
        print(f"错误: HR目录 {hr_dir_path} 不存在或不是一个目录.")
        return
    if not lr_dir_path.is_dir():
        print(f"错误: LR目录 {lr_dir_path} 不存在或不是一个目录.")
        return

    # 获取所有图像对
    image_pairs = get_image_pairs(hr_dir_path, lr_dir_path, lr_suffix, img_ext)
    if not image_pairs:
        print("未找到图像对, 脚本终止.")
        return

    # 打乱图像对顺序
    random.shuffle(image_pairs)

    # 创建输出目录
    if output_dir.exists():
        print(f"警告: 输出目录 {output_dir} 已存在. 内容可能被覆盖.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 分割并复制文件
    num_total_pairs = len(image_pairs)
    current_idx = 0

    for split_name, ratio in split_config:
        num_split_pairs = round(num_total_pairs * ratio)
        
        # 确保最后一个分割取走所有剩余的，以处理舍入问题
        if split_name == split_config[-1][0]:
            split_pairs = image_pairs[current_idx:]
        else:
            split_pairs = image_pairs[current_idx : current_idx + num_split_pairs]
        
        if not split_pairs:
            print(f"警告: 分割 '{split_name}' 的图像数量为0.")
            continue

        print(f"为分割 '{split_name}' 分配 {len(split_pairs)} 对图像.")

        output_split_dir = output_dir / split_name
        output_hr_split_dir = output_split_dir / hr_subdir_name
        output_lr_split_dir = output_split_dir / lr_subdir_name

        output_hr_split_dir.mkdir(parents=True, exist_ok=True)
        output_lr_split_dir.mkdir(parents=True, exist_ok=True)

        for hr_path, lr_path in split_pairs:
            try:
                shutil.copy(hr_path, output_hr_split_dir / hr_path.name)
                shutil.copy(lr_path, output_lr_split_dir / lr_path.name)
            except Exception as e:
                print(f"错误: 复制文件时出错 HR: {hr_path}, LR: {lr_path}. 错误: {e}")


        current_idx += len(split_pairs)

    print(f"\n数据集分割完成. 分割后的数据保存在: {output_dir.resolve()}")
    
    # 验证所有文件是否已分配
    if current_idx < num_total_pairs:
        print(f"警告: {num_total_pairs - current_idx} 个图像对由于舍入问题未被分配到任何分割中。请检查分割比例。")
    elif current_idx > num_total_pairs:
         print(f"警告: 分配的图像对总数 ({current_idx}) 超过了可用总数 ({num_total_pairs})。这可能是一个bug。")


if __name__ == "__main__":
    main()
