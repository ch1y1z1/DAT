import torch
import yaml
from collections import OrderedDict
from basicsr.archs.dat_arch import DAT # Make sure DAT is importable

def analyze_model_parameters(model):
    """
    Analyzes and prints the number of parameters for each module in the model.

    Args:
        model (torch.nn.Module): The model to analyze.
    """
    print(f"{'Module':<70} {'Parameters':<15} {'Percentage (%)':<15}")
    print("-" * 100)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params == 0:
        print("No trainable parameters found in the model.")
        return

    for name, module in model.named_modules():
        module_params = [p for p in module.parameters(recurse=False) if p.requires_grad]
        if module_params:
            params = sum(p.numel() for p in module_params)
            if params > 0:
                percentage = 100 * params / total_params
                print(f"{name:<70} {params:<15,d} {percentage:<15.2f}")

    print("-" * 100)
    print(f"{'Total trainable parameters:':<70} {total_params:<15,d}")

if __name__ == '__main__':
    config_path = 'train_DAT_light_x4_val.yml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        exit(1)

    network_g_config = config.get('network_g')
    if not network_g_config:
        print(f"Error: 'network_g' not found in {config_path}")
        exit(1)

    try:
        model = DAT(
            upscale=network_g_config.get('upscale', 4),
            in_chans=network_g_config.get('in_chans', 3),
            img_size=network_g_config.get('img_size', 64),
            img_range=network_g_config.get('img_range', 1.),
            depth=network_g_config.get('depth', [18]),
            embed_dim=network_g_config.get('embed_dim', 36),
            num_heads=network_g_config.get('num_heads', [3]),
            expansion_factor=network_g_config.get('expansion_factor', 1.5),
            resi_connection=network_g_config.get('resi_connection', '3conv'),
            split_size=network_g_config.get('split_size', [8, 16]),
            upsampler=network_g_config.get('upsampler', 'pixelshuffledirect'),
            rank_ratio=network_g_config.get('rank_ratio', 0.5),
            high_rank_ratio=network_g_config.get('high_rank_ratio',1.0),
            high_rank_depth_threshold=network_g_config.get('high_rank_depth_threshold',2)
        )
    except Exception as e:
        print(f"Error instantiating DAT model: {e}")
        print("Please ensure all required parameters for DAT are defined in the YAML and match the class definition.")
        exit(1)

    weights_path = './net_g.pth'
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'network_g' in checkpoint :
             state_dict = checkpoint['network_g']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
        
        transformed_state_dict = new_state_dict.copy()
        keys_to_delete = []
        keys_to_add = OrderedDict()

        candidate_prefixes = []
        for module_name_full, module_instance in model.named_modules():
            if isinstance(module_instance, torch.nn.Linear):
                has_U_in_checkpoint = (module_name_full + ".U.weight") in new_state_dict
                has_V_in_checkpoint = (module_name_full + ".V.weight") in new_state_dict
                if has_U_in_checkpoint and has_V_in_checkpoint:
                    # Check if this specific module path was already chosen with a '.qkv_proj' suffix strategy (avoid double add)
                    already_targeted = any(module_name_full.startswith(cp_chosen) for cp_chosen, _ in candidate_prefixes)
                    if not already_targeted:
                         candidate_prefixes.append((module_name_full, module_name_full)) # model_path, checkpoint_path_prefix
                else: # Try with .qkv_proj suffix for checkpoint keys
                    checkpoint_candidate_path = module_name_full + ".qkv_proj"
                    has_U_qkv_proj = (checkpoint_candidate_path + ".U.weight") in new_state_dict
                    has_V_qkv_proj = (checkpoint_candidate_path + ".V.weight") in new_state_dict
                    if has_U_qkv_proj and has_V_qkv_proj:
                        already_targeted = any(module_name_full.startswith(cp_chosen) for cp_chosen, _ in candidate_prefixes)
                        if not already_targeted:
                            candidate_prefixes.append((module_name_full, checkpoint_candidate_path))
        
        for model_path_prefix, ckpt_path_prefix in candidate_prefixes:
            print(f"INFO: Attempting to convert low-rank keys from checkpoint prefix '{ckpt_path_prefix}' for model module '{model_path_prefix}'")
            W_U_key = ckpt_path_prefix + ".U.weight"
            W_V_key = ckpt_path_prefix + ".V.weight"

            if W_U_key not in new_state_dict or W_V_key not in new_state_dict:
                print(f"WARN: Missing U or V weights in checkpoint for prefix {ckpt_path_prefix}. Skipping this conversion.")
                continue

            W_U = new_state_dict[W_U_key]
            W_V = new_state_dict[W_V_key]
            
            reconstructed_weight = W_V @ W_U
            keys_to_add[model_path_prefix + ".weight"] = reconstructed_weight
            keys_to_delete.extend([W_U_key, W_V_key])
            
            # Bias handling (assuming bias, if present, is with V)
            module_instance = model.get_submodule(model_path_prefix)
            if module_instance.bias is not None:
                V_bias_key = ckpt_path_prefix + ".V.bias"
                if V_bias_key in new_state_dict:
                    bias_V = new_state_dict[V_bias_key]
                    keys_to_add[model_path_prefix + ".bias"] = bias_V
                    keys_to_delete.append(V_bias_key)
                # else: model expects bias, but not found in checkpoint under this prefix. load_state_dict(strict=False) will handle.
            else: # Model's Linear layer does NOT expect a bias, remove if present in checkpoint under this prefix
                for b_key_suffix in [".V.bias", ".U.bias", ".bias"]: #.bias for non-decomposed bias that might be there
                    b_key = ckpt_path_prefix + b_key_suffix
                    if b_key in new_state_dict:
                        keys_to_delete.append(b_key)

        unique_keys_to_delete = list(set(keys_to_delete)) # Ensure unique keys for deletion
        for k_del in unique_keys_to_delete:
            if k_del in transformed_state_dict:
                del transformed_state_dict[k_del]
        transformed_state_dict.update(keys_to_add)
        new_state_dict = transformed_state_dict

        print("INFO: Attempting to load weights. Strict mode is OFF. Mismatches will be ignored.")
        print("INFO: The parameter analysis will reflect the model structure defined in the YAML.")
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully attempted to load weights from {weights_path} (non-strictly).")

    except FileNotFoundError:
        print(f"Error: Weights file not found at {weights_path}")
        exit(1)
    except RuntimeError as e:
        print(f"RuntimeError during weight loading (even with strict=False): {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading weights: {e}")
        exit(1)

    model.eval()
    
    print("\nModel Parameter Analysis (based on YAML-defined structure):")
    analyze_model_parameters(model) 