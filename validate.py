import torch
from basicsr.models import build_model
from basicsr.utils.options import parse_options
import os.path as osp
from fvcore.nn import FlopCountAnalysis


def evaluate(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    model = build_model(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_fake = torch.rand(1, 3, 256, 256).to(device)
    
    # 使用 model.net_g 而不是 model，因为 net_g 是实际的 PyTorch 网络模型
    network = model.net_g
    # 将网络设置为评估模式以避免 BatchNorm 的问题
    network.eval()
    
    flops = FlopCountAnalysis(network, input_fake).total()
    flops = flops / 10**9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))
    
    # 计算参数数量也使用 net_g
    num_parameters = sum(p.numel() for p in network.parameters())
    num_parameters = num_parameters / 10**6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    evaluate(root_path)
