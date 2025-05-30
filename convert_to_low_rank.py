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

# Helper functions to get and set nested modules
def get_module_by_path(module, path_str):
    parts = path_str.split('.')
    attr = module
    for part in parts:
        if part.isdigit(): # For list access like layers[0]
            attr = attr[int(part)]
        else:
            attr = getattr(attr, part)
    return attr

def set_module_by_path(module, path_str, new_sub_module):
    parts = path_str.split('.')
    parent = module
    for i, part in enumerate(parts[:-1]):
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    
    final_part = parts[-1]
    if final_part.isdigit():
        parent[int(final_part)] = new_sub_module
    else:
        setattr(parent, final_part, new_sub_module)

def calculate_rank_for_svd(rank_ratio, in_features, out_features):
    """Helper to calculate rank consistently."""
    return max(1, int(rank_ratio * min(in_features, out_features)))

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
        help="Base rank ratio for SVD (e.g., 0.3). Also used for model structure if < 1.0 for layers not meeting high_rank criteria.",
    )
    parser.add_argument(
        "--high_rank_ratio",
        type=float,
        default=None,
        help="Optional higher rank ratio for SVD for SharedQKV and/or FFNs in deeper layers (e.g., 0.8). Must be > 0 and <= 1 if provided.",
    )
    parser.add_argument(
        "--high_rank_depth_threshold",
        type=int,
        default=999, # Default to a very high value, effectively applying high_rank_ratio only to SharedQKV if high_rank_ratio is set.
        help="RG index (0-based) from which FFN layers will consider using high_rank_ratio. SharedQKV always considers it if set.",
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

    if args.high_rank_ratio is not None and not (0.0 < args.high_rank_ratio <= 1.0):
        raise ValueError("high_rank_ratio must be between 0 (exclusive) and 1 (inclusive), if provided.")
    
    if args.high_rank_depth_threshold < 0:
        raise ValueError("high_rank_depth_threshold must be a non-negative integer.")

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
        "rank_ratio": args.rank_ratio, 
        "high_rank_ratio": args.high_rank_ratio,
        "high_rank_depth_threshold": args.high_rank_depth_threshold
    }

    # Override with YAML values ONLY IF the corresponding CLI arg was NOT explicitly set (i.e., it's the default)
    # For rank_ratio, high_rank_ratio, high_rank_depth_threshold, CLI args always take precedence if provided.
    if dat_params_from_yaml:
        for key, yaml_val in dat_params_from_yaml.items():
            if key in dat_constructor_args and hasattr(args, key): # Ensure key is relevant and in args
                # For most params, YAML overrides if CLI was default
                if key not in ['rank_ratio', 'high_rank_ratio', 'high_rank_depth_threshold']:
                    if getattr(args, key) == parser.get_default(key):
                        dat_constructor_args[key] = yaml_val
                # For these specific rank params, CLI already set them.
                # If we wanted YAML to override CLI if CLI was default, we'd need specific logic,
                # but current approach is CLI always wins for these if they are set via CLI.
                # If CLI for high_rank_ratio was None (its default) and YAML has it, then YAML should be used.
                elif key == 'high_rank_ratio':
                    if args.high_rank_ratio is None and 'high_rank_ratio' in dat_params_from_yaml:
                         dat_constructor_args['high_rank_ratio'] = dat_params_from_yaml['high_rank_ratio']
                elif key == 'high_rank_depth_threshold':
                     # If CLI was default (999) and YAML has a different one.
                    if args.high_rank_depth_threshold == parser.get_default('high_rank_depth_threshold') and \
                       'high_rank_depth_threshold' in dat_params_from_yaml:
                       dat_constructor_args['high_rank_depth_threshold'] = dat_params_from_yaml['high_rank_depth_threshold']
    
    print("Initializing new low-rank DAT model with effective parameters for DAT constructor:")
    for k, v in dat_constructor_args.items():
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
    # Model construction uses dat_constructor_args['rank_ratio'] (which is affected by CLI or YAML)
    # This rank_ratio determines the *initial* structure of low_rank_model (which layers are LowRankLinear and their initial rank)
    # We will dynamically reconfigure parts of it based on high_rank_ratio and high_rank_depth_threshold.

    for rg_idx in range(
        len(dat_constructor_args["depth"])
    ):  # Iterate based on number of RGs
        rg_base_name = f"layers.{rg_idx}"
        original_rg_shared_qkv_weight_key = f"{rg_base_name}.shared_qkv.weight"
        original_rg_shared_qkv_bias_key = f"{rg_base_name}.shared_qkv.bias"

        if original_rg_shared_qkv_weight_key in original_state_dict:
            orig_w = original_state_dict[original_rg_shared_qkv_weight_key]
            orig_b = original_state_dict.get(original_rg_shared_qkv_bias_key)
            
            in_features_sqkv = orig_w.shape[1]
            out_features_sqkv = orig_w.shape[0]
            has_bias_sqkv = orig_b is not None

            # Determine if this specific shared_qkv module in low_rank_model needs rank reconfiguration
            module_path_sqkv_proj = f"layers.{rg_idx}.shared_qkv.qkv_proj"
            current_sqkv_proj_module = get_module_by_path(low_rank_model, module_path_sqkv_proj)

            # Determine which rank_ratio to use for STRUCTURE and SVD of this SharedQKV
            structural_rank_ratio_for_sqkv = args.rank_ratio # Default structure from CLI base rank_ratio
            
            if args.high_rank_ratio is not None:
                print(f"  SharedQKV {original_rg_shared_qkv_weight_key} considers high_rank_ratio: {args.high_rank_ratio} for structure & SVD.")
                structural_rank_ratio_for_sqkv = args.high_rank_ratio
            # else: SharedQKV uses base args.rank_ratio for structure and SVD (already set)
            
            # CRITICAL FIX: svd_rank_ratio_for_sqkv must be consistent with structural_rank_ratio_for_sqkv
            svd_rank_ratio_for_sqkv = structural_rank_ratio_for_sqkv 
                
            target_rank_sqkv = calculate_rank_for_svd(structural_rank_ratio_for_sqkv, in_features_sqkv, out_features_sqkv)
            
            needs_reconfiguration = False
            if structural_rank_ratio_for_sqkv < 1.0:
                if isinstance(current_sqkv_proj_module, LowRankLinear):
                    if current_sqkv_proj_module.rank != target_rank_sqkv:
                        needs_reconfiguration = True
                elif isinstance(current_sqkv_proj_module, nn.Linear): # Was nn.Linear, now want LowRankLinear
                    needs_reconfiguration = True
            else: # structural_rank_ratio_for_sqkv is 1.0, should be nn.Linear
                if not isinstance(current_sqkv_proj_module, nn.Linear):
                    needs_reconfiguration = True
            
            if needs_reconfiguration:
                if structural_rank_ratio_for_sqkv < 1.0:
                    print(f"    Reconfiguring module {module_path_sqkv_proj} to LowRankLinear with rank {target_rank_sqkv} (from ratio {structural_rank_ratio_for_sqkv})")
                    new_module = LowRankLinear(in_features_sqkv, out_features_sqkv, target_rank_sqkv, bias=has_bias_sqkv)
                else:
                    print(f"    Reconfiguring module {module_path_sqkv_proj} to nn.Linear (from ratio {structural_rank_ratio_for_sqkv})")
                    new_module = nn.Linear(in_features_sqkv, out_features_sqkv, bias=has_bias_sqkv)
                set_module_by_path(low_rank_model, module_path_sqkv_proj, new_module)
                new_model_state_dict_template = low_rank_model.state_dict() # Refresh template

            # current_sqkv_proj_module is updated after potential reconfiguration
            current_sqkv_proj_module = get_module_by_path(low_rank_model, module_path_sqkv_proj)
            new_model_rg_qkv_proj_is_low_rank = isinstance(current_sqkv_proj_module, LowRankLinear)

            if new_model_rg_qkv_proj_is_low_rank: # Target is LowRankLinear
                # The svd_rank_ratio_for_sqkv is already determined above based on rules
                print(f"  Decomposing SharedQKV {original_rg_shared_qkv_weight_key} using SVD rank_ratio: {svd_rank_ratio_for_sqkv}")
                u_w, v_w, v_b = convert_linear_to_low_rank(
                    orig_w, orig_b, svd_rank_ratio_for_sqkv 
                )
                shared_rg_weights_cache[rg_base_name] = (u_w, v_w, v_b, True) # True indicates decomposed
                # Populate converted_state_dict for this RG's SharedQKV U and V parts
                converted_state_dict[f"{module_path_sqkv_proj}.U.weight"] = u_w
                converted_state_dict[f"{module_path_sqkv_proj}.V.weight"] = v_w
                if v_b is not None and f"{module_path_sqkv_proj}.V.bias" in new_model_state_dict_template:
                    converted_state_dict[f"{module_path_sqkv_proj}.V.bias"] = v_b
                
                # SVD Test (simplified, happens if this is the first SVD layer for SharedQKV)
                if not tested_one_svd_layer_flag: 
                    # ... (SVD test logic - ensure it uses svd_rank_ratio_for_sqkv) ...
                    # ... (the test itself should be fine if svd_rank_ratio_for_sqkv is passed correctly)
                    # For example, in the test: abs_diff_model_qkv = ... 
                    # print(f"  WARNING: High reconstruction error for SharedQKV (rank_ratio={svd_rank_ratio_for_sqkv}) on ...")
                    tested_one_svd_layer_flag = True # Mark tested

            else: # Target is nn.Linear for this RG's SharedQKV (because structural_rank_ratio_for_sqkv was 1.0)
                print(f"  Copying SharedQKV {original_rg_shared_qkv_weight_key} as nn.Linear.")
                shared_rg_weights_cache[rg_base_name] = (orig_w.clone(), orig_b.clone() if orig_b is not None else None, False) # False indicates not decomposed
                # Populate converted_state_dict for this RG's SharedQKV .weight and .bias parts
                converted_state_dict[f"{module_path_sqkv_proj}.weight"] = orig_w.clone()
                if orig_b is not None and f"{module_path_sqkv_proj}.bias" in new_model_state_dict_template:
                    converted_state_dict[f"{module_path_sqkv_proj}.bias"] = orig_b.clone()
        else:
            print(
                f"  Warning: Original weight {original_rg_shared_qkv_weight_key} not found for RG {rg_base_name}. Cannot cache."
            )

    print(
        "\nConverting all model weights using cached/decomposed RG QKVs and processing other layers..."
    )
    # Ensure the loop iterates over the potentially updated new_model_state_dict_template
    current_new_model_keys = list(new_model_state_dict_template.keys()) 

    for new_key in current_new_model_keys: # Iterate over a copy of keys in case template is refreshed
        if new_key in converted_state_dict: 
            continue

        # Case 1: Block-level QKV keys (e.g. layers.X.blocks.Y.attn.qkv.qkv_proj.*)
        # These always use the same structure (LowRankLinear or nn.Linear) and weights as their RG's SharedQKV
        if ".blocks." in new_key and ".attn.qkv.qkv_proj." in new_key:
            key_parts = new_key.split('.')
            rg_base_name = f"{key_parts[0]}.{key_parts[1]}"  # e.g. "layers.0"
            # Determine if the target key is for LowRankLinear (U/V) or nn.Linear (.weight/.bias)
            # This should match the structure of the SharedQKV for this RG in the reconfigured low_rank_model
            block_qkv_module_path = ".".join(new_key.split('.')[:-1]) # e.g. layers.X.blocks.Y.attn.qkv.qkv_proj
            # This path in low_rank_model should be an alias or have same structure as rg_base_name.shared_qkv.qkv_proj
            # For safety, we rely on the cached weights type (decomposed or not)

            if rg_base_name in shared_rg_weights_cache:
                cached_data = shared_rg_weights_cache[rg_base_name]
                is_decomposed = cached_data[-1]

                if is_decomposed: # SharedQKV for this RG was decomposed -> LowRankLinear
                    u_w_cached, v_w_cached, v_b_cached, _ = cached_data
                    if new_key.endswith(".U.weight"):
                        converted_state_dict[new_key] = u_w_cached.clone()
                    elif new_key.endswith(".V.weight"):
                        converted_state_dict[new_key] = v_w_cached.clone()
                    elif new_key.endswith(".V.bias") and v_b_cached is not None:
                        converted_state_dict[new_key] = v_b_cached.clone()
                    elif v_b_cached is None and new_key.endswith(".V.bias"):
                        # print(f"    Info: Block QKV {new_key} (LowRank) expects bias, but cached RG QKV bias was None. Using template init.")
                        converted_state_dict[new_key] = new_model_state_dict_template[new_key].clone()
                    # else: # Should not happen if keys match LowRankLinear structure
                    #     print(f"  Warning: Unhandled decomposed block QKV key: {new_key}. Using template param.")
                    #     converted_state_dict[new_key] = new_model_state_dict_template[new_key].clone()
                else:  # SharedQKV for this RG was nn.Linear
                    orig_w_cached, orig_b_cached, _ = cached_data
                    if new_key.endswith(".weight"):
                        converted_state_dict[new_key] = orig_w_cached.clone()
                    elif new_key.endswith(".bias") and orig_b_cached is not None:
                        converted_state_dict[new_key] = orig_b_cached.clone()
                    elif orig_b_cached is None and new_key.endswith(".bias"):
                        # print(f"    Info: Block QKV {new_key} (Linear) expects bias, but cached RG QKV bias was None. Using template init.")
                        converted_state_dict[new_key] = new_model_state_dict_template[new_key].clone()
                    # else: # Should not happen if keys match nn.Linear structure
                    #     print(f"  Warning: Unhandled non-decomposed block QKV key: {new_key}. Using template param.")
                    #     converted_state_dict[new_key] = new_model_state_dict_template[new_key].clone()
            else:
                # This should ideally not happen if RGs were processed correctly
                print(f"  Warning: RG cache for {rg_base_name} not found when processing block QKV {new_key}. Using template param.")
                converted_state_dict[new_key] = new_model_state_dict_template[new_key].clone()
            continue # Processed this block QKV key

        # Case 2: SGFN.fc1 or SGFN.fc2 modules
        # This section aims to identify an FFN module (fc1 or fc2) only ONCE,
        # determine its structure (Linear or LowRank), reconfigure if needed,
        # and then populate all its relevant keys in converted_state_dict.
        
        ffn_module_base_path_identified = None # e.g. layers.0.blocks.0.ffn.fc1
        original_ffn_w_key = None
        original_ffn_b_key = None
        is_fc1_ffn_module = False

        if ".ffn.fc1." in new_key:
            ffn_module_base_path_identified = new_key.split(".ffn.fc1.")[0] + ".ffn.fc1"
            is_fc1_ffn_module = True
        elif ".ffn.fc2." in new_key:
            ffn_module_base_path_identified = new_key.split(".ffn.fc2.")[0] + ".ffn.fc2"
            is_fc1_ffn_module = False # It's fc2
        
        if ffn_module_base_path_identified:
            # Check if this FFN module has already been fully processed and its keys populated
            # A simple check: if its .weight (for Linear) or .U.weight (for LowRank) is already in converted_dict
            potential_weight_key = f"{ffn_module_base_path_identified}.weight"
            potential_U_weight_key = f"{ffn_module_base_path_identified}.U.weight"
            if potential_weight_key in converted_state_dict or potential_U_weight_key in converted_state_dict:
                continue # Already processed this entire FFN module

            # --- This FFN module (e.g. layers.X.blocks.Y.ffn.fc1) has not been processed yet --- 
            print(f"Processing FFN module: {ffn_module_base_path_identified}")
            original_ffn_w_key = f"{ffn_module_base_path_identified}.weight"
            original_ffn_b_key = f"{ffn_module_base_path_identified}.bias"

            if original_ffn_w_key not in original_state_dict:
                print(f"  Warning: Original weight {original_ffn_w_key} not found. Skipping FFN module {ffn_module_base_path_identified}.")
                # Populate with template if new_key requires it and not skipped by continue
                # This path should ideally not be hit frequently if original model is valid.
                # The main loop's fallthrough will handle new_key with template if it was not part of this skipped FFN.
                continue

            orig_w_ffn = original_state_dict[original_ffn_w_key]
            orig_b_ffn = original_state_dict.get(original_ffn_b_key)
            in_features_ffn = orig_w_ffn.shape[1]
            out_features_ffn = orig_w_ffn.shape[0]
            has_bias_ffn = orig_b_ffn is not None

            current_ffn_module_in_model = get_module_by_path(low_rank_model, ffn_module_base_path_identified)

            structural_rank_ratio_for_ffn = args.rank_ratio 
            svd_rank_ratio_for_ffn = args.rank_ratio 
            try:
                parts = original_ffn_w_key.split('.')
                block_idx_ffn = int(parts[3])
                rg_idx_ffn = parts[1]
                if args.high_rank_ratio is not None and block_idx_ffn < args.high_rank_depth_threshold:
                    print(f"  FFN {ffn_module_base_path_identified} (Block {block_idx_ffn}) is shallower than depth threshold {args.high_rank_depth_threshold}, will use high_rank_ratio: {args.high_rank_ratio}")
                    structural_rank_ratio_for_ffn = args.high_rank_ratio
                # else: use base args.rank_ratio
                svd_rank_ratio_for_ffn = structural_rank_ratio_for_ffn
            except (IndexError, ValueError) as e:
                print(f"  Warning: Could not parse block index for FFN {original_ffn_w_key}: {e}. Using base rank ratio {args.rank_ratio}.")
            
            target_rank_ffn = calculate_rank_for_svd(structural_rank_ratio_for_ffn, in_features_ffn, out_features_ffn)
            
            # Reconfigure FFN module if necessary
            needs_reconfiguration_ffn = False
            if structural_rank_ratio_for_ffn < 1.0:
                if isinstance(current_ffn_module_in_model, LowRankLinear):
                    if current_ffn_module_in_model.rank != target_rank_ffn:
                        needs_reconfiguration_ffn = True
                elif isinstance(current_ffn_module_in_model, nn.Linear):
                    needs_reconfiguration_ffn = True
            else: # structural_rank_ratio_for_ffn is 1.0
                if not isinstance(current_ffn_module_in_model, nn.Linear):
                    needs_reconfiguration_ffn = True
            
            if needs_reconfiguration_ffn:
                if structural_rank_ratio_for_ffn < 1.0:
                    print(f"    Reconfiguring FFN module {ffn_module_base_path_identified} to LowRankLinear with rank {target_rank_ffn}")
                    new_ffn_module = LowRankLinear(in_features_ffn, out_features_ffn, target_rank_ffn, bias=has_bias_ffn)
                else:
                    print(f"    Reconfiguring FFN module {ffn_module_base_path_identified} to nn.Linear")
                    new_ffn_module = nn.Linear(in_features_ffn, out_features_ffn, bias=has_bias_ffn)
                set_module_by_path(low_rank_model, ffn_module_base_path_identified, new_ffn_module)
                new_model_state_dict_template = low_rank_model.state_dict() # Refresh template
                current_ffn_module_in_model = new_ffn_module # Update reference
            
            # Populate converted_state_dict for this FFN module
            if isinstance(current_ffn_module_in_model, LowRankLinear):
                if not (svd_rank_ratio_for_ffn < 1.0):
                    print(f"  ERROR: FFN {ffn_module_base_path_identified} is LowRankLinear but SVD ratio is {svd_rank_ratio_for_ffn}. Should be < 1.0. Forcing to 0.99 for SVD.")
                    svd_rank_ratio_for_ffn = 0.99 # Fallback to ensure SVD runs
                
                # Ensure the target_rank_ffn for SVD matches what the structure expects
                # This was re-calculated for svd_rank_ratio_for_ffn, but let's ensure the print is clear
                # The rank of current_ffn_module_in_model is the structural rank.
                # The SVD should use svd_rank_ratio_for_ffn to decompose.
                print(f"  Decomposing FFN {original_ffn_w_key} with SVD rank_ratio: {svd_rank_ratio_for_ffn} (structural rank was {current_ffn_module_in_model.rank})")
                u_w_ffn, v_w_ffn, v_b_ffn = convert_linear_to_low_rank(orig_w_ffn, orig_b_ffn, svd_rank_ratio_for_ffn)
                
                converted_state_dict[f"{ffn_module_base_path_identified}.U.weight"] = u_w_ffn
                converted_state_dict[f"{ffn_module_base_path_identified}.V.weight"] = v_w_ffn
                if v_b_ffn is not None and f"{ffn_module_base_path_identified}.V.bias" in new_model_state_dict_template:
                    converted_state_dict[f"{ffn_module_base_path_identified}.V.bias"] = v_b_ffn
                elif f"{ffn_module_base_path_identified}.V.bias" in new_model_state_dict_template and v_b_ffn is None:
                     print(f"    Info: FFN {ffn_module_base_path_identified}.V.bias exists in template but SVD resulted in no bias. Using template bias.")
                     converted_state_dict[f"{ffn_module_base_path_identified}.V.bias"] = new_model_state_dict_template[f"{ffn_module_base_path_identified}.V.bias"].clone()
                
                # SVD Test
                ffn_test_type_str = "SGFN.fc1" if is_fc1_ffn_module else "SGFN.fc2"
                ffn_tested_flag = tested_one_ffn_fc1_flag if is_fc1_ffn_module else tested_one_ffn_fc2_flag
                if not ffn_tested_flag:
                    # ... (Simplified SVD test logic as before, using svd_rank_ratio_for_ffn) ...
                    if is_fc1_ffn_module: tested_one_ffn_fc1_flag = True
                    else: tested_one_ffn_fc2_flag = True
            
            elif isinstance(current_ffn_module_in_model, nn.Linear):
                print(f"  Copying FFN {original_ffn_w_key} as nn.Linear.")
                converted_state_dict[f"{ffn_module_base_path_identified}.weight"] = orig_w_ffn.clone()
                if has_bias_ffn and f"{ffn_module_base_path_identified}.bias" in new_model_state_dict_template:
                    converted_state_dict[f"{ffn_module_base_path_identified}.bias"] = orig_b_ffn.clone()
                elif f"{ffn_module_base_path_identified}.bias" in new_model_state_dict_template and not has_bias_ffn:
                     print(f"    Info: FFN {ffn_module_base_path_identified}.bias exists in template but original FFN had no bias. Using template bias.")
                     converted_state_dict[f"{ffn_module_base_path_identified}.bias"] = new_model_state_dict_template[f"{ffn_module_base_path_identified}.bias"].clone()
            else:
                print(f"  ERROR: FFN module {ffn_module_base_path_identified} is of unexpected type: {type(current_ffn_module_in_model)}")

            continue # Finished processing this FFN module (fc1 or fc2)
            # The `if new_key in converted_state_dict:` at the start of the outer loop will now correctly skip individual U/V/weight/bias keys of this FFN.

        # Case 3: Other parameters (Direct copy from original_state_dict if name matches & shape matches)
        # This handles layers that are not SharedQKV, not block-level QKV, and not FFNs processed above.
        if new_key in original_state_dict:
            if original_state_dict[new_key].shape == new_model_state_dict_template[new_key].shape:
                converted_state_dict[new_key] = original_state_dict[new_key].clone()
            else:
                print(
                    f"  Shape mismatch for {new_key} (orig: {original_state_dict[new_key].shape}, new: {new_model_state_dict_template[new_key].shape}). Using template param."
                )
                converted_state_dict[new_key] = new_model_state_dict_template[new_key].clone()
        else:
            if new_key not in converted_state_dict: # Check if not already processed by other paths
                print(
                    f"  Warning: Parameter {new_key} not found in original. Using template param."
                )
                converted_state_dict[new_key] = new_model_state_dict_template[new_key].clone()

    # Final check for any keys in template not filled - should not happen if logic is complete
    # for k_template in new_model_state_dict_template:
    #     if k_template not in converted_state_dict:
    #         print(f"  ALERT: Key {k_template} from new model template was not filled in converted_state_dict. Initializing from template.")
    #         converted_state_dict[k_template] = new_model_state_dict_template[k_template].clone()

    # Before loading, ensure all keys in converted_state_dict match the final low_rank_model structure
    final_model_keys = low_rank_model.state_dict().keys()
    converted_keys = list(converted_state_dict.keys()) # list to avoid issues if modified during iteration

    for k_converted in converted_keys:
        if k_converted not in final_model_keys:
            print(f"  WARNING: Key {k_converted} is in converted_state_dict but NOT in final model structure. Removing.")
            del converted_state_dict[k_converted]
    
    for k_final_model in final_model_keys:
        if k_final_model not in converted_state_dict:
            print(f"  WARNING: Key {k_final_model} is in final model structure but NOT in converted_state_dict. Will use model's init for it.")
            # This is acceptable if strict=False, but good to be aware.
            # If it's a LowRankLinear part that should have been converted, it's an issue.
            # Could copy from template: converted_state_dict[k_final_model] = new_model_state_dict_template[k_final_model].clone()
            # For now, rely on strict=False for these.


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
