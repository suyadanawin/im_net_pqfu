import torch
import numpy as np

def get_actual_weights(path):
    """Extracts the state_dict regardless of how it's wrapped."""
    ckpt = torch.load(path, map_location='cpu')
    # Based on your logs, the weights are inside 'model_state_dict'
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    # Clean prefixes
    new_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('model.', '')
        new_dict[name] = v
    return new_dict

def compare(name_a, dict_a, name_b, dict_b):
    total_l2 = 0.0
    layer_sims = []
    
    print(f"\n--- Comparing {name_a} vs {name_b} ---")
    print(f"{'Layer Name':<40} | {'L2 Dist':<10} | {'Cos Sim':<8}")
    
    for name, w_b in dict_b.items():
        if name in dict_a and "weight" in name and w_b.ndim > 0:
            w_a = dict_a[name].float().view(-1)
            w_b = w_b.float().view(-1)
            
            if w_a.shape == w_b.shape:
                dist = torch.norm(w_a - w_b, p=2).item()
                total_l2 += dist**2
                cos_sim = torch.nn.functional.cosine_similarity(w_a, w_b, dim=0).item()
                layer_sims.append(cos_sim)
                # Print only first few and last few layers to keep output clean
                if len(layer_sims) < 5 or "fc" in name:
                    print(f"{name[:40]:<40} | {dist:<10.4f} | {cos_sim:<8.4f}")

    if layer_sims:
        print(f"OVERALL L2: {np.sqrt(total_l2):.4f}")
        print(f"AVG COSINE: {np.mean(layer_sims):.4f}")
    else:
        print("No matching layers found. Check key names.")

# Load all three for a complete research comparison
try:
    weights_oracle = get_actual_weights('./outputs_phase3/checkpoints/oracle_final.pt')
    weights_unlearn = get_actual_weights('./outputs_phase3/checkpoints/unlearn_final.pt')
    weights_full = get_actual_weights('./outputs_phase2/checkpoints/global_best.pt')

    compare("Unlearn", weights_unlearn, "Oracle", weights_oracle)
    compare("Full Model", weights_full, "Oracle", weights_oracle)
except FileNotFoundError as e:
    print(f"Error: {e}")