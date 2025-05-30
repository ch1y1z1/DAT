import torch
import torch.nn as nn
from collections import OrderedDict
import argparse
import sys
import os
import yaml  # Added for YAML parsing
from typing import Optional  # Added for older Python compatibility

# Ensure basicsr can be found if the script is not in the project root,
# or if basicsr is not in PYTHONPATH.
# Assuming the script is in the root of the DAT project.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from basicsr.archs.dat_arch import DAT, LowRankLinear
except ImportError:
    print("Error: Could not import DAT model from basicsr.archs.dat_arch.")
    print(
        "Please ensure that the script is in the correct directory and basicsr is in your PYTHONPATH."
    )
    sys.exit(1)


def convert_linear_to_low_rank(
    linear_layer_weight: torch.Tensor,
    linear_layer_bias: Optional[torch.Tensor],
    rank_ratio: float,
):
    """
    Decomposes a linear layer's weights into U and V for LowRankLinear.
    This version distributes sqrt(S) to both U and V matrices.

    Args:
        linear_layer_weight (torch.Tensor): Weight tensor of the original nn.Linear layer [out_features, in_features].
        linear_layer_bias (Optional[torch.Tensor]): Bias tensor of the original nn.Linear layer [out_features], or None.
        rank_ratio (float): Ratio to determine the rank.

    Returns:
        tuple: (u_weight, v_weight, v_bias)
               u_weight: Weight for LowRankLinear.U [rank, in_features]
               v_weight: Weight for LowRankLinear.V [out_features, rank]
               v_bias: Bias for LowRankLinear.V [out_features], or None
    """
    W = linear_layer_weight.data.float()  # Ensure float for SVD

    U_svd, S_svd_vec, Vt_svd = torch.linalg.svd(W, full_matrices=False)

    in_features = W.shape[1]
    out_features = W.shape[0]

    rank = max(1, int(rank_ratio * min(in_features, out_features)))
    if rank > S_svd_vec.size(0):
        rank = S_svd_vec.size(0)

    U_r = U_svd[:, :rank]
    S_r_vec = S_svd_vec[:rank]
    Vt_r = Vt_svd[:rank, :]

    # Distribute sqrt(S) to both U and V
    # W_approx = (U_r @ diag(sqrt(S_r_vec))) @ (diag(sqrt(S_r_vec)) @ Vt_r)
    # LowRankLinear.U.weight (rank, in_features) will be diag(sqrt(S_r_vec)) @ Vt_r
    # LowRankLinear.V.weight (out_features, rank) will be U_r @ diag(sqrt(S_r_vec))

    S_r_sqrt_diag = torch.diag(torch.sqrt(S_r_vec))

    u_weight = (
        S_r_sqrt_diag @ Vt_r
    )  # Shape: (rank, rank) @ (rank, in_features) -> (rank, in_features)
    v_weight = (
        U_r @ S_r_sqrt_diag
    )  # Shape: (out_features, rank) @ (rank, rank) -> (out_features, rank)

    v_bias = linear_layer_bias.data.clone() if linear_layer_bias is not None else None

    return u_weight, v_weight, v_bias


def main():
    parser = argparse.ArgumentParser(
        description="Convert a DAT model to a Low-Rank DAT model."
    )
    parser.add_argument(
        "--original_model_path",
        type=str,
        required=True,
        help="Path to the original pre-trained DAT model (.pth file).",
    )
    parser.add_argument(
        "--converted_model_path",
        type=str,
        required=True,
        help="Path to save the converted low-rank DAT model (.pth file).",
    )
    parser.add_argument(
        "--rank_ratio",
        type=float,
        required=True,
        help="Rank ratio for SVD (e.g., 0.5 for 50% rank). Must be > 0 and <= 1.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to a YAML configuration file for DAT model parameters.",
    )

    # DAT model constructor arguments - These can be overridden by YAML or command line.
    # Defaults should ideally match DAT constructor defaults if not specified elsewhere.
    parser.add_argument("--img_size", type=int, default=64, help="Input image size.")
    parser.add_argument(
        "--in_chans", type=int, default=3, help="Number of input image channels."
    )
    parser.add_argument(
        "--embed_dim", type=int, default=180, help="Patch embedding dimension."
    )
    # Default for split_size in DAT constructor is [2,4], but typical usage (and example YAML) is [8,16] or [8,8]
    # The DAT model internally handles using [8,8] for the first two RGs regardless of this param.
    parser.add_argument(
        "--split_size",
        type=int,
        nargs=2,
        default=[8, 16],
        metavar=("H_SPLIT", "W_SPLIT"),
        help="Height and Width of spatial window for later RGs.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        nargs="+",
        default=[2, 2, 2, 2],
        help="Depth of each residual group.",
    )  # DAT default: [2,2,2,2]
    parser.add_argument(
        "--num_heads",
        type=int,
        nargs="+",
        default=[2, 2, 2, 2],
        help="Number of attention heads in different RGs.",
    )  # DAT default: [2,2,2,2]
    parser.add_argument(
        "--expansion_factor",
        type=float,
        default=4.0,
        help="Ratio of FFN hidden dim to embedding dim.",
    )  # DAT default: 4.
    parser.add_argument(
        "--qkv_bias",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="If True, add a learnable bias to query, key, value.",
    )  # DAT default: True
    parser.add_argument(
        "--qk_scale",
        type=float,
        default=None,
        help="Override default qk scale of head_dim ** -0.5 if set.",
    )  # DAT default: None
    parser.add_argument(
        "--drop_rate", type=float, default=0.0, help="Dropout rate."
    )  # DAT default: 0.
    parser.add_argument(
        "--attn_drop_rate", type=float, default=0.0, help="Attention dropout rate."
    )  # DAT default: 0.
    parser.add_argument(
        "--drop_path_rate", type=float, default=0.1, help="Stochastic depth rate."
    )  # DAT default: 0.1
    parser.add_argument(
        "--upscale", type=int, default=2, help="Upscale factor."
    )  # DAT default: 2
    parser.add_argument(
        "--img_range", type=float, default=1.0, help="Image range."
    )  # DAT default: 1.
    parser.add_argument(
        "--resi_connection",
        type=str,
        default="1conv",
        choices=["1conv", "3conv"],
        help="Type of residual connection block.",
    )  # DAT default: '1conv'
    parser.add_argument(
        "--upsampler",
        type=str,
        default="pixelshuffle",
        choices=["pixelshuffle", "pixelshuffledirect"],
        help="Type of upsampler.",
    )  # DAT default: 'pixelshuffle'

    args = parser.parse_args()

    if not (0.0 < args.rank_ratio <= 1.0):
        raise ValueError("rank_ratio must be between 0 (exclusive) and 1 (inclusive).")

    # Load parameters from YAML if config_path is provided
    # Command-line arguments will override YAML values if both are present and CLI is not default.
    dat_params_from_yaml = {}
    if args.config_path:
        print(f"Loading DAT parameters from YAML: {args.config_path}")
        try:
            with open(args.config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
            if "network_g" in yaml_config:
                dat_params_from_yaml = yaml_config["network_g"]
                print(
                    f"  Loaded parameters from network_g section: {dat_params_from_yaml}"
                )
            else:
                print(
                    f"  Warning: 'network_g' section not found in {args.config_path}. Using command-line/default DAT parameters."
                )
        except FileNotFoundError:
            print(
                f"  Error: YAML config file not found at {args.config_path}. Using command-line/default DAT parameters."
            )
            sys.exit(1)
        except yaml.YAMLError as e:
            print(
                f"  Error parsing YAML file {args.config_path}: {e}. Using command-line/default DAT parameters."
            )
            sys.exit(1)

    # Consolidate parameters: CLI > YAML > ArgParse Defaults
    # The `args` object already has CLI values or ArgParse defaults.
    # We update `args` if a YAML value exists and the CLI value was the default.

    dat_constructor_args = {
        "img_size": args.img_size,
        "in_chans": args.in_chans,
        "embed_dim": args.embed_dim,
        "split_size": args.split_size,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "expansion_factor": args.expansion_factor,
        "qkv_bias": args.qkv_bias,
        "qk_scale": args.qk_scale,
        "drop_rate": args.drop_rate,
        "attn_drop_rate": args.attn_drop_rate,
        "drop_path_rate": args.drop_path_rate,
        "upscale": args.upscale,
        "img_range": args.img_range,
        "resi_connection": args.resi_connection,
        "upsampler": args.upsampler,
        "rank_ratio": args.rank_ratio,  # This is for the new model, not from YAML
    }

    # Override with YAML values if the corresponding CLI arg was not explicitly set (i.e., it's the default)
    if dat_params_from_yaml:
        for key, yaml_val in dat_params_from_yaml.items():
            if hasattr(args, key):
                if getattr(args, key) == parser.get_default(key):
                    dat_constructor_args[key] = yaml_val

    print("Initializing new low-rank DAT model with effective parameters:")
    for k, v in dat_constructor_args.items():
        if k != "rank_ratio":
            print(f"  {k}: {v}")

    low_rank_model = DAT(**dat_constructor_args)

    # Verification of QKV sharing - REMOVED FOR SIMPLICITY
    # print("---- Verifying QKV sharing ----")
    # ... (entire verification block removed) ...
    # print("---- End Verification ----")
    new_model_state_dict_template = low_rank_model.state_dict()

    print(f"Loading original model state_dict from {args.original_model_path}...")
    try:
        ckpt = torch.load(
            args.original_model_path, map_location="cpu", weights_only=False
        )  # Set weights_only based on trust
    except Exception as e:
        print(f"Error loading original model: {e}")
        try:
            print("Attempting to load with weights_only=True...")
            ckpt = torch.load(
                args.original_model_path, map_location="cpu", weights_only=True
            )
        except Exception as e2:
            print(f"Error loading original model even with weights_only=True: {e2}")
            sys.exit(1)

    if "params_ema" in ckpt:
        original_state_dict = ckpt["params_ema"]
    elif "params" in ckpt:
        original_state_dict = ckpt["params"]
    else:
        original_state_dict = ckpt

    converted_state_dict = OrderedDict()
    # Cache for RG-level weights (decomposed or original)
    # Key: rg_base_name (e.g., "layers.0"),
    # Value: tuple (u_w, v_w, v_b, is_decomposed_flag) or (orig_w, orig_b, is_decomposed_flag)
    shared_rg_weights_cache = {}
    tested_one_svd_layer_flag = False  # For SharedQKV in RG
    tested_one_ffn_fc1_flag = False  # For SGFN.fc1
    tested_one_ffn_fc2_flag = False  # For SGFN.fc2

    print(
        "Processing and Caching Residual Group shared QKV weights first..."
    )  # Keep high-level
    # Determine effective rank_ratio for model construction vs SVD
    # Model construction uses dat_constructor_args['rank_ratio'] (which is affected by CLI)
    # SVD decomposition itself should use args.rank_ratio from CLI for consistency in this script's context
    svd_rank_ratio_to_use = args.rank_ratio

    for rg_idx in range(
        len(dat_constructor_args["depth"])
    ):  # Iterate based on number of RGs
        rg_base_name = f"layers.{rg_idx}"
        original_rg_shared_qkv_weight_key = f"{rg_base_name}.shared_qkv.weight"
        original_rg_shared_qkv_bias_key = f"{rg_base_name}.shared_qkv.bias"

        if original_rg_shared_qkv_weight_key in original_state_dict:
            orig_w = original_state_dict[original_rg_shared_qkv_weight_key]
            orig_b = original_state_dict.get(original_rg_shared_qkv_bias_key)

            # Check if the corresponding new model part is LowRankLinear or nn.Linear
            # This depends on dat_constructor_args['rank_ratio'] used for DAT() init
            new_model_rg_qkv_proj_is_low_rank = False
            if 0.0 < dat_constructor_args["rank_ratio"] < 1.0:
                # Check if the key for LowRankLinear U part exists in the template for this RG
                if (
                    f"{rg_base_name}.shared_qkv.qkv_proj.U.weight"
                    in new_model_state_dict_template
                ):
                    new_model_rg_qkv_proj_is_low_rank = True

            if (
                new_model_rg_qkv_proj_is_low_rank
            ):  # New model has LowRankLinear for this RG's SharedQKV
                # print( # REMOVED
                #     f"  Decomposing RG {rg_base_name} shared QKV: {original_rg_shared_qkv_weight_key} for LowRankLinear."
                # )
                u_w, v_w, v_b = convert_linear_to_low_rank(
                    orig_w, orig_b, svd_rank_ratio_to_use
                )
                shared_rg_weights_cache[rg_base_name] = (u_w, v_w, v_b, True)
                # Directly populate the RG-level LowRankLinear params in converted_state_dict
                converted_state_dict[f"{rg_base_name}.shared_qkv.qkv_proj.U.weight"] = (
                    u_w
                )
                converted_state_dict[f"{rg_base_name}.shared_qkv.qkv_proj.V.weight"] = (
                    v_w
                )
                if (
                    v_b is not None
                    and f"{rg_base_name}.shared_qkv.qkv_proj.V.bias"
                    in new_model_state_dict_template
                ):
                    converted_state_dict[
                        f"{rg_base_name}.shared_qkv.qkv_proj.V.bias"
                    ] = v_b

                if (
                    not tested_one_svd_layer_flag
                ):  # Perform SVD test for the first decomposed RG QKV
                    # Simplified test: only print WARNING if model's rank error is high
                    # print( # REMOVED
                    #     f"\\n[DEBUG] Testing single layer SVD reconstruction for SharedQKV: {original_rg_shared_qkv_weight_key}"
                    # )
                    u_w_test_full_qkv, v_w_test_full_qkv, v_b_test_full_qkv = (
                        convert_linear_to_low_rank(
                            orig_w.clone(),
                            orig_b.clone() if orig_b is not None else None,
                            1.0,  # Force full rank for this specific test comparison
                        )
                    )

                    in_f_orig_qkv, out_f_orig_qkv = orig_w.shape[1], orig_w.shape[0]
                    original_linear_layer_qkv = nn.Linear(
                        in_f_orig_qkv, out_f_orig_qkv, bias=(orig_b is not None)
                    )
                    original_linear_layer_qkv.weight.data = orig_w.clone()
                    if orig_b is not None:
                        original_linear_layer_qkv.bias.data = orig_b.clone()

                    # 1. Test with actual rank used for the model (using u_w, v_w, v_b from the main SVD)
                    rank_val_model_qkv = u_w.shape[0]
                    reconstructed_lr_layer_model_qkv = LowRankLinear(
                        in_features=in_f_orig_qkv,
                        out_features=out_f_orig_qkv,
                        rank=rank_val_model_qkv,
                        bias=(v_b is not None),
                    )
                    reconstructed_lr_layer_model_qkv.U.weight.data = u_w.clone()
                    reconstructed_lr_layer_model_qkv.V.weight.data = v_w.clone()
                    if (
                        v_b is not None
                        and reconstructed_lr_layer_model_qkv.V.bias is not None
                    ):
                        reconstructed_lr_layer_model_qkv.V.bias.data = v_b.clone()

                    # 2. Test with full rank decomposition (using u_w_test_full_qkv, v_w_test_full_qkv, v_b_test_full_qkv)
                    rank_val_full_qkv = u_w_test_full_qkv.shape[0]
                    reconstructed_lr_layer_full_rank_qkv = LowRankLinear(
                        in_features=in_f_orig_qkv,
                        out_features=out_f_orig_qkv,
                        rank=rank_val_full_qkv,
                        bias=(v_b_test_full_qkv is not None),
                    )
                    reconstructed_lr_layer_full_rank_qkv.U.weight.data = (
                        u_w_test_full_qkv.clone()
                    )
                    reconstructed_lr_layer_full_rank_qkv.V.weight.data = (
                        v_w_test_full_qkv.clone()
                    )
                    if (
                        v_b_test_full_qkv is not None
                        and reconstructed_lr_layer_full_rank_qkv.V.bias is not None
                    ):
                        reconstructed_lr_layer_full_rank_qkv.V.bias.data = (
                            v_b_test_full_qkv.clone()
                        )

                    test_input_qkv = torch.randn(1, 2, in_f_orig_qkv).float()
                    original_linear_layer_qkv.eval()
                    reconstructed_lr_layer_model_qkv.eval()
                    reconstructed_lr_layer_full_rank_qkv.eval()

                    with torch.no_grad():
                        y_orig_qkv = original_linear_layer_qkv(test_input_qkv)
                        y_model_approx_qkv = reconstructed_lr_layer_model_qkv(
                            test_input_qkv
                        )
                        y_full_rank_approx_qkv = reconstructed_lr_layer_full_rank_qkv(
                            test_input_qkv
                        )

                    # Simplified print logic below
                    # print(f"  Input shape for SharedQKV test: {test_input_qkv.shape}") # REMOVED
                    # print( # REMOVED
                    #     "  --- Comparing Original vs. Model's Rank Approx (SharedQKV, rank_ratio={}) ---".format(
                    #         svd_rank_ratio_to_use
                    #     )
                    # )
                    abs_diff_model_qkv = (y_orig_qkv - y_model_approx_qkv).abs()
                    # print(f"    Rank used for model's LowRankLinear: {rank_val_model_qkv}") # REMOVED
                    # print(f"    Mean Absolute Difference (model vs orig): {abs_diff_model_qkv.mean().item():.6e}") # REMOVED
                    # print(f"    Mean Relative Difference (model vs orig): {(abs_diff_model_qkv / (torch.abs(y_orig_qkv) + 1e-9)).mean().item():.6e}") # REMOVED
                    # print(f"    Norm original: {torch.norm(y_orig_qkv).item():.4f}, Norm model approx: {torch.norm(y_model_approx_qkv).item():.4f}") # REMOVED

                    # print( # REMOVED
                    #     "  --- Comparing Original vs. Full Rank SVD Approx (SharedQKV, rank_ratio=1.0 for test) ---"
                    # )
                    abs_diff_full_qkv = (y_orig_qkv - y_full_rank_approx_qkv).abs()
                    # print(f"    Rank used for full rank test's LowRankLinear: {rank_val_full_qkv}") # REMOVED
                    # print(f"    Mean Absolute Difference (full rank test vs orig): {abs_diff_full_qkv.mean().item():.6e}") # REMOVED
                    # print(f"    Mean Relative Difference (full rank test vs orig): {(abs_diff_full_qkv / (torch.abs(y_orig_qkv) + 1e-9)).mean().item():.6e}") # REMOVED
                    # print(f"    Norm original: {torch.norm(y_orig_qkv).item():.4f}, Norm full rank SVD approx: {torch.norm(y_full_rank_approx_qkv).item():.4f}") # REMOVED

                    if (  # Check full rank SVD integrity
                        abs_diff_full_qkv.mean().item() > 1e-5
                        or (
                            abs(
                                torch.norm(y_orig_qkv)
                                - torch.norm(y_full_rank_approx_qkv)
                            )
                            / (torch.norm(y_orig_qkv) + 1e-9)
                        )
                        > 1e-4
                    ):
                        print(
                            f"  WARNING: High error in FULL RANK SVD reconstruction for SharedQKV: {original_rg_shared_qkv_weight_key}. This might indicate an SVD logic issue."
                        )
                    # else: # REMOVED "seems OK" print
                    # print(
                    #     f"    [DEBUG] SharedQKV Full rank SVD reconstruction through LowRankLinear seems OK."
                    # )

                    if (  # Check model's rank SVD error
                        abs_diff_model_qkv.mean().item() > 5e-3
                        or (
                            abs(torch.norm(y_orig_qkv) - torch.norm(y_model_approx_qkv))
                            / (torch.norm(y_orig_qkv) + 1e-9)
                        )
                        > 0.05
                    ):
                        print(
                            f"  WARNING: High reconstruction error for SharedQKV (rank_ratio={svd_rank_ratio_to_use}) on {original_rg_shared_qkv_weight_key}. MAE: {abs_diff_model_qkv.mean().item():.2e}"
                        )
                    # else: # REMOVED "seems OK" print
                    # print(
                    #     f"    [DEBUG] SharedQKV Model's rank ({svd_rank_ratio_to_use}) SVD reconstruction seems OK for {original_rg_shared_qkv_weight_key}."
                    # )
                    # print("[DEBUG] End of extended single SharedQKV SVD layer test.\\n") # REMOVED
                    tested_one_svd_layer_flag = True
            else:  # New model has standard nn.Linear for this RG's SharedQKV
                # print( # REMOVED
                #     f"  Caching original RG {rg_base_name} shared QKV: {original_rg_shared_qkv_weight_key} for nn.Linear."
                # )
                shared_rg_weights_cache[rg_base_name] = (
                    orig_w.clone(),
                    orig_b.clone() if orig_b is not None else None,
                    False,
                )
                # Directly populate the RG-level nn.Linear params in converted_state_dict
                converted_state_dict[f"{rg_base_name}.shared_qkv.qkv_proj.weight"] = (
                    orig_w.clone()
                )
                if (
                    orig_b is not None
                    and f"{rg_base_name}.shared_qkv.qkv_proj.bias"
                    in new_model_state_dict_template
                ):
                    converted_state_dict[f"{rg_base_name}.shared_qkv.qkv_proj.bias"] = (
                        orig_b.clone()
                    )
        else:
            print(
                f"  Warning: Original weight {original_rg_shared_qkv_weight_key} not found for RG {rg_base_name}. Cannot cache."
            )

    print(
        "\nConverting all model weights using cached/decomposed RG QKVs and processing other layers..."
    )
    for new_key, template_param in new_model_state_dict_template.items():
        if (
            new_key in converted_state_dict
        ):  # Already handled by RG caching pass (e.g. layers.X.shared_qkv.qkv_proj.*)
            continue

        # Case 1: Block-level QKV keys (e.g. layers.X.blocks.Y.attn.qkv.qkv_proj.*)
        if ".blocks." in new_key and ".attn.qkv.qkv_proj." in new_key:
            key_parts = new_key.split(".")
            rg_base_name = f"{key_parts[0]}.{key_parts[1]}"  # e.g. "layers.0"

            if rg_base_name in shared_rg_weights_cache:
                cached_data = shared_rg_weights_cache[rg_base_name]
                is_decomposed = cached_data[-1]

                if is_decomposed:
                    u_w_cached, v_w_cached, v_b_cached, _ = cached_data
                    if ".U.weight" in new_key:
                        converted_state_dict[new_key] = u_w_cached.clone()
                    elif ".V.weight" in new_key:
                        converted_state_dict[new_key] = v_w_cached.clone()
                    elif ".V.bias" in new_key and v_b_cached is not None:
                        converted_state_dict[new_key] = v_b_cached.clone()
                    elif v_b_cached is None and ".V.bias" in new_key:
                        # Keep this info print as it might be useful if bias is unexpectedly missing
                        print(
                            f"    Info: Block QKV {new_key} expects bias, but cached RG QKV bias was None. Using template init."
                        )
                        converted_state_dict[new_key] = template_param.clone()
                    else:  # Keep this warning
                        print(
                            f"  Warning: Unhandled decomposed block QKV key: {new_key}. Using template param."
                        )
                        converted_state_dict[new_key] = template_param.clone()
                else:  # Cached weights are original (not decomposed)
                    orig_w_cached, orig_b_cached, _ = cached_data
                    if new_key.endswith(
                        ".weight"
                    ):  # e.g. layers.X.blocks.Y.attn.qkv.qkv_proj.weight
                        converted_state_dict[new_key] = orig_w_cached.clone()
                    elif new_key.endswith(".bias") and orig_b_cached is not None:
                        converted_state_dict[new_key] = orig_b_cached.clone()
                    elif orig_b_cached is None and new_key.endswith(".bias"):
                        # Keep this info print
                        print(
                            f"    Info: Block QKV {new_key} expects bias, but cached RG QKV bias was None. Using template init."
                        )
                        converted_state_dict[new_key] = template_param.clone()
                    else:  # Keep this warning
                        print(
                            f"  Warning: Unhandled non-decomposed block QKV key: {new_key}. Using template param."
                        )
                        converted_state_dict[new_key] = template_param.clone()
            else:
                print(
                    f"  Warning: RG cache for {rg_base_name} not found when processing block QKV {new_key}. Using template param."
                )
                converted_state_dict[new_key] = template_param.clone()
            continue

        # Case 2: SGFN.fc1 or SGFN.fc2
        # Determine if SGFN layers in the new model are LowRankLinear or nn.Linear
        # This depends on dat_constructor_args['rank_ratio'] used for DAT() init
        new_model_ffn_is_low_rank = False
        if 0.0 < dat_constructor_args["rank_ratio"] < 1.0:
            # Check if template key matches LowRankLinear structure for FFNs
            if ".ffn.fc1.U.weight" in new_key or ".ffn.fc2.U.weight" in new_key:
                new_model_ffn_is_low_rank = True

        if new_model_ffn_is_low_rank:
            ffn_base_key = ""
            original_ffn_w_key = ""
            original_ffn_b_key = ""
            if ".ffn.fc1.U.weight" in new_key:
                ffn_base_key = new_key.split(".ffn.fc1.U.weight")[0]
                original_ffn_w_key = f"{ffn_base_key}.ffn.fc1.weight"
                original_ffn_b_key = f"{ffn_base_key}.ffn.fc1.bias"
            elif ".ffn.fc2.U.weight" in new_key:
                ffn_base_key = new_key.split(".ffn.fc2.U.weight")[0]
                original_ffn_w_key = f"{ffn_base_key}.ffn.fc2.weight"
                original_ffn_b_key = f"{ffn_base_key}.ffn.fc2.bias"
            # Handle V.weight and V.bias for FFN LowRankLinear
            elif any(
                s in new_key
                for s in [
                    ".ffn.fc1.V.weight",
                    ".ffn.fc1.V.bias",
                    ".ffn.fc2.V.weight",
                    ".ffn.fc2.V.bias",
                ]
            ):
                # These are covered when U.weight is processed for the same FFN layer
                if (
                    new_key not in converted_state_dict
                ):  # Safety check, should be populated already
                    print(
                        f"  Warning: FFN V-part key {new_key} was not populated by U-part. Using template."
                    )
                    converted_state_dict[new_key] = template_param.clone()
                continue

            if original_ffn_w_key and original_ffn_w_key in original_state_dict:
                print(f"  Decomposing FFN layer {original_ffn_w_key} for {new_key}")
                orig_w_ffn = original_state_dict[original_ffn_w_key]
                orig_b_ffn = original_state_dict.get(original_ffn_b_key)
                u_w_ffn, v_w_ffn, v_b_ffn = convert_linear_to_low_rank(
                    orig_w_ffn, orig_b_ffn, svd_rank_ratio_to_use
                )
                converted_state_dict[new_key] = u_w_ffn  # For U.weight
                v_weight_key = new_key.replace(".U.weight", ".V.weight")
                v_bias_key = new_key.replace(".U.weight", ".V.bias")
                converted_state_dict[v_weight_key] = v_w_ffn
                if v_b_ffn is not None and v_bias_key in new_model_state_dict_template:
                    converted_state_dict[v_bias_key] = v_b_ffn

                # Determine if it's fc1 or fc2 for flagging the test
                is_fc1 = ".ffn.fc1.U.weight" in new_key
                current_ffn_tested_flag = (
                    tested_one_ffn_fc1_flag if is_fc1 else tested_one_ffn_fc2_flag
                )
                ffn_type_str = "SGFN.fc1" if is_fc1 else "SGFN.fc2"

                if not current_ffn_tested_flag:
                    # Simplified test: only print WARNING if model's rank error is high
                    # print( # REMOVED
                    #     f"\\n[DEBUG] Testing single layer SVD reconstruction for {ffn_type_str}: {original_ffn_w_key}"
                    # )
                    u_w_test_full_ffn, v_w_test_full_ffn, v_b_test_full_ffn = (
                        convert_linear_to_low_rank(
                            orig_w_ffn.clone(),
                            orig_b_ffn.clone() if orig_b_ffn is not None else None,
                            1.0,  # Force full rank for this test
                        )
                    )
                    in_f_orig_ffn, out_f_orig_ffn = (
                        orig_w_ffn.shape[1],
                        orig_w_ffn.shape[0],
                    )
                    original_linear_layer_ffn = nn.Linear(
                        in_f_orig_ffn, out_f_orig_ffn, bias=(orig_b_ffn is not None)
                    )
                    original_linear_layer_ffn.weight.data = orig_w_ffn.clone()
                    if orig_b_ffn is not None:
                        original_linear_layer_ffn.bias.data = orig_b_ffn.clone()

                    rank_val_model_ffn = u_w_ffn.shape[0]
                    reconstructed_lr_layer_model_ffn = LowRankLinear(
                        in_features=in_f_orig_ffn,
                        out_features=out_f_orig_ffn,
                        rank=rank_val_model_ffn,
                        bias=(v_b_ffn is not None),
                    )
                    reconstructed_lr_layer_model_ffn.U.weight.data = u_w_ffn.clone()
                    reconstructed_lr_layer_model_ffn.V.weight.data = v_w_ffn.clone()
                    if (
                        v_b_ffn is not None
                        and reconstructed_lr_layer_model_ffn.V.bias is not None
                    ):
                        reconstructed_lr_layer_model_ffn.V.bias.data = v_b_ffn.clone()

                    test_input_ffn = torch.randn(1, 2, in_f_orig_ffn).float()
                    original_linear_layer_ffn.eval()
                    reconstructed_lr_layer_model_ffn.eval()
                    reconstructed_lr_layer_full_rank_ffn = LowRankLinear(
                        in_features=in_f_orig_ffn,
                        out_features=out_f_orig_ffn,
                        rank=rank_val_model_ffn,
                        bias=(v_b_test_full_ffn is not None),
                    )
                    reconstructed_lr_layer_full_rank_ffn.U.weight.data = (
                        u_w_test_full_ffn.clone()
                    )
                    reconstructed_lr_layer_full_rank_ffn.V.weight.data = (
                        v_w_test_full_ffn.clone()
                    )
                    if (
                        v_b_test_full_ffn is not None
                        and reconstructed_lr_layer_full_rank_ffn.V.bias is not None
                    ):
                        reconstructed_lr_layer_full_rank_ffn.V.bias.data = (
                            v_b_test_full_ffn.clone()
                        )

                    with torch.no_grad():
                        y_orig_ffn = original_linear_layer_ffn(test_input_ffn)
                        y_model_approx_ffn = reconstructed_lr_layer_model_ffn(
                            test_input_ffn
                        )
                        y_full_rank_approx_ffn = reconstructed_lr_layer_full_rank_ffn(
                            test_input_ffn
                        )

                    # print(f"  Input shape for {ffn_type_str} test: {test_input_ffn.shape}") # REMOVED
                    # print( # REMOVED
                    #     "  --- Comparing Original vs. Model's Rank Approx ({}, rank_ratio={}) ---".format(
                    #         ffn_type_str, svd_rank_ratio_to_use
                    #     )
                    # )
                    abs_diff_model_ffn = (y_orig_ffn - y_model_approx_ffn).abs()
                    # print(f"    Rank used for model's LowRankLinear: {rank_val_model_ffn}") # REMOVED
                    # print(f"    Mean Absolute Difference (model vs orig): {abs_diff_model_ffn.mean().item():.6e}") # REMOVED
                    # print(f"    Mean Relative Difference (model vs orig): {(abs_diff_model_ffn / (torch.abs(y_orig_ffn) + 1e-9)).mean().item():.6e}") # REMOVED
                    # print(f"    Norm original: {torch.norm(y_orig_ffn).item():.4f}, Norm model approx: {torch.norm(y_model_approx_ffn).item():.4f}") # REMOVED

                    # print( # REMOVED
                    #     "  --- Comparing Original vs. Full Rank SVD Approx ({}, rank_ratio=1.0 for test) ---".format(
                    #         ffn_type_str
                    #     )
                    # )
                    abs_diff_full_ffn = (y_orig_ffn - y_full_rank_approx_ffn).abs()
                    # print(f"    Rank used for full rank test's LowRankLinear: {rank_val_full_ffn}") # REMOVED
                    # print(f"    Mean Absolute Difference (full rank test vs orig): {abs_diff_full_ffn.mean().item():.6e}") # REMOVED
                    # print(f"    Mean Relative Difference (full rank test vs orig): {(abs_diff_full_ffn / (torch.abs(y_orig_ffn) + 1e-9)).mean().item():.6e}") # REMOVED
                    # print(f"    Norm original: {torch.norm(y_orig_ffn).item():.4f}, Norm full rank SVD approx: {torch.norm(y_full_rank_approx_ffn).item():.4f}") # REMOVED

                    if (  # Check full rank SVD integrity
                        abs_diff_full_ffn.mean().item() > 1e-5
                        or (
                            abs(
                                torch.norm(y_orig_ffn)
                                - torch.norm(y_full_rank_approx_ffn)
                            )
                            / (torch.norm(y_orig_ffn) + 1e-9)
                        )
                        > 1e-4
                    ):
                        print(
                            f"  WARNING: High error in FULL RANK SVD reconstruction for {ffn_type_str}: {original_ffn_w_key}. This might indicate an SVD logic issue."
                        )
                    # else: # REMOVED "seems OK" print
                    # print(
                    #     f"    [DEBUG] {ffn_type_str} Full rank SVD reconstruction through LowRankLinear seems OK."
                    # )

                    if (  # Check model's rank SVD error
                        abs_diff_model_ffn.mean().item() > 5e-3
                        or (
                            abs(torch.norm(y_orig_ffn) - torch.norm(y_model_approx_ffn))
                            / (torch.norm(y_orig_ffn) + 1e-9)
                        )
                        > 0.05
                    ):
                        print(
                            f"  WARNING: High reconstruction error for {ffn_type_str} (rank_ratio={svd_rank_ratio_to_use}) on {original_ffn_w_key}. MAE: {abs_diff_model_ffn.mean().item():.2e}"
                        )
                    # else: # REMOVED "seems OK" print
                    # print(
                    #     f"    [DEBUG] {ffn_type_str} Model's rank ({svd_rank_ratio_to_use}) SVD reconstruction seems OK for {original_ffn_w_key}."
                    # )
                    # print(f"[DEBUG] End of extended single {ffn_type_str} SVD layer test.\\n") # REMOVED

                    if is_fc1:
                        tested_one_ffn_fc1_flag = True
                    else:
                        tested_one_ffn_fc2_flag = True

            elif original_ffn_w_key:  # Keep this warning
                print(
                    f"  Warning: Original weight {original_ffn_w_key} not found for FFN SVD. Using template for {new_key} and its V-parts."
                )
                converted_state_dict[new_key] = template_param.clone()
                # Also initialize V parts if U was initialized
                v_weight_key = new_key.replace(".U.weight", ".V.weight")
                v_bias_key = new_key.replace(".U.weight", ".V.bias")
                if v_weight_key in new_model_state_dict_template:
                    converted_state_dict[v_weight_key] = new_model_state_dict_template[
                        v_weight_key
                    ].clone()
                if v_bias_key in new_model_state_dict_template:
                    converted_state_dict[v_bias_key] = new_model_state_dict_template[
                        v_bias_key
                    ].clone()
            # No continue here if original_ffn_w_key was empty, fall through to general copy for other FFN keys if any
            if original_ffn_w_key:  # If we identified an FFN U.weight key, we've handled it and its V parts.
                continue

        # Case 3: Other parameters (Direct copy, or nn.Linear FFNs if not LowRankLinear)
        original_key_to_find = new_key  # Default direct mapping

        # Check for nn.Linear SGFN fc1/fc2 if new_model_ffn_is_low_rank is False
        if not new_model_ffn_is_low_rank:
            if (
                ".ffn.fc1.weight" in new_key
                or ".ffn.fc1.bias" in new_key
                or ".ffn.fc2.weight" in new_key
                or ".ffn.fc2.bias" in new_key
            ):
                # original_key_to_find is already new_key, which is correct for direct copy from original nn.Linear FFN
                pass  # Handled by fallthrough direct copy

        # General copy logic
        if original_key_to_find in original_state_dict:
            if original_state_dict[original_key_to_find].shape == template_param.shape:
                converted_state_dict[new_key] = original_state_dict[
                    original_key_to_find
                ].clone()
            else:
                print(
                    f"  Shape mismatch for {new_key} (orig: {original_state_dict[original_key_to_find].shape}, new: {template_param.shape}). Using template param."
                )
                converted_state_dict[new_key] = template_param.clone()
        else:
            if (
                new_key not in converted_state_dict
            ):  # Check if not already processed (e.g. by RG cache pass or FFN V-parts)
                # Keep this warning
                print(
                    f"  Warning: Parameter {original_key_to_find} (mapped from {new_key}) not found in original. Using template param."
                )
                converted_state_dict[new_key] = template_param.clone()

    low_rank_model.load_state_dict(converted_state_dict, strict=False)
    print(
        "Attempted to load converted state_dict into the new low-rank model."
    )  # Keep high-level

    save_content = {}
    if "params_ema" in ckpt or "params" in ckpt:
        for k, v in ckpt.items():
            if k == "params_ema":
                save_content["params_ema"] = low_rank_model.state_dict()
            elif k == "params":
                save_content["params"] = low_rank_model.state_dict()
            else:
                save_content[k] = v
    else:
        save_content = low_rank_model.state_dict()

    torch.save(save_content, args.converted_model_path)
    print(
        f"Successfully converted model saved to {args.converted_model_path}"
    )  # Keep high-level


if __name__ == "__main__":
    main()
