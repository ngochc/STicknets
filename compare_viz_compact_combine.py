#!/usr/bin/env python3

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.STickNet import build_STickNet
from models.TickNet import build_TickNet
from util import get_device

warnings.filterwarnings('ignore')

DEVICE = get_device()

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

class FRPDPMultiModelHook:
    def __init__(self):
        self.model_features = {}
        self.hooks = {}

    def hook_fn(self, model_name, layer_name):
        def hook(module, input, output):
            if model_name not in self.model_features:
                self.model_features[model_name] = {}
            self.model_features[model_name][layer_name] = output.detach().cpu()
        return hook

    def register_fr_pdp_hooks(self, models_dict):
        print("Registering FR-PDP hooks across all models...")
        for model_name, model in models_dict.items():
            print(f"Processing {model_name} model...")
            self.hooks[model_name] = []
            found_blocks = []
            for name, module in model.named_modules():
                if module.__class__.__name__ == 'FR_PDP_block':
                    found_blocks.append(name)
                    hook = module.register_forward_hook(self.hook_fn(model_name, name))
                    self.hooks[model_name].append(hook)
                    print(f"Hooked FR_PDP_block: {name}")
            print(f"{model_name}: {len(found_blocks)} FR-PDP blocks hooked")
            if len(found_blocks) == 0:
                print(f"No FR_PDP blocks found in {model_name}!")
                for name, module in list(model.named_modules())[:10]:
                    print(f"{name}: {type(module).__name__}")
            else:
                print(f"Hooked modules: {found_blocks}")

    def extract_features(self, models_dict, input_images):
        print("Extracting FR-PDP features from all models...")
        for model_name, model in models_dict.items():
            print(f"Processing {model_name}...")
            with torch.no_grad():
                _ = model(input_images)
        print("Feature extraction completed!")
        return self.model_features

    def remove_all_hooks(self):
        for model_name, model_hooks in self.hooks.items():
            for hook in model_hooks:
                hook.remove()
        self.hooks.clear()
        self.model_features.clear()

def load_target_image(image_path, image_size=(224, 224), batch_size=16):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target image not found: {image_path}")
        print(f"Loading target image: {image_path}")
        original_image = Image.open(image_path).convert('RGB')
        images = []
        for _ in range(batch_size):
            image_tensor = transform(original_image)
            images.append(image_tensor)
        batch_images = torch.stack(images).to(DEVICE)
        print(f"Image loaded successfully! Shape: {batch_images.shape}")
        return batch_images
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Creating synthetic fallback data...")
        synthetic_data = torch.randn(batch_size, 3, *image_size).to(DEVICE)
        return synthetic_data

def load_models():
    models = {}
    print("Loading model variants...")

    model_defs = [
        (
            "STickNet_basic", 
            build_STickNet(
                num_classes=120,
                typesize='basic',
                cifar=False,
                use_lightweight_optimization=False
            ),
            "/checkpoint/STickNet_basic.pth"
        ),
        (
            "STickNet_basic_star", 
            build_STickNet(
                num_classes=120,
                typesize='basic',
                cifar=False,
                use_lightweight_optimization=True
            ),
            "/checkpoint/STickNet_basic_star.pth"
        ),
        (
            "TickNet_basic",
            build_TickNet(
                num_classes=120,
                typesize='basic',
                cifar=False
            ),
            "/checkpoint/TickNet_basic.pth"
        ),
    ]

    for name, model, checkpoint in model_defs:
        if checkpoint and os.path.isfile(checkpoint):
            print(f"Loading checkpoint for {name} from {checkpoint}")
            state = torch.load(checkpoint, map_location=DEVICE)
            if "state_dict" in state:
                state = state["state_dict"]
            # Remove 'module.' prefix if present
            new_state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
            model.load_state_dict(new_state, strict=False)
        else:
            print(f"No valid checkpoint for {name}, using randomly initialized weights.")
        model = model.to(DEVICE)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        models[name] = model
        print(f"{name}: {total_params:,} parameters")

    return models

def get_pdp_id_from_module_name(module_name):
    if 'stage2.unit' in module_name:
        return "PDP1"
    elif 'stage3.unit' in module_name:
        return "PDP2"
    elif 'stage4.unit' in module_name:
        return "PDP3"
    else:
        return "PDP_Other"

def organize_fr_pdp_features(model_features):
    organized = {}
    for model_name, model_layers in model_features.items():
        organized[model_name] = {}
        for layer_name, features in model_layers.items():
            pdp_id = get_pdp_id_from_module_name(layer_name)
            if pdp_id not in ["PDP_Other"]:
                if pdp_id not in organized[model_name]:
                    organized[model_name][pdp_id] = []
                organized[model_name][pdp_id].append(features)
        for pdp_id in organized[model_name]:
            if organized[model_name][pdp_id]:
                organized[model_name][pdp_id] = organized[model_name][pdp_id][0]
    return organized

def apply_all_combinations(feature_maps):
    if len(feature_maps.shape) == 3:
        feature_maps = feature_maps.unsqueeze(0)
    combinations = {}
    combinations['Mean'] = torch.mean(feature_maps, dim=1, keepdim=True)
    combinations['Max'], _ = torch.max(feature_maps, dim=1, keepdim=True)
    combinations['RMS'] = torch.sqrt(torch.mean(feature_maps**2, dim=1, keepdim=True))
    combinations['Sum'] = torch.sum(feature_maps, dim=1, keepdim=True)
    combinations['Variance'] = torch.var(feature_maps, dim=1, keepdim=True)
    channel_importance = torch.mean(torch.abs(feature_maps), dim=(2, 3), keepdim=True)
    attention_weights = F.softmax(channel_importance, dim=1)
    combinations['Attention'] = torch.sum(feature_maps * attention_weights, dim=1, keepdim=True)
    combinations['L2_Norm'] = torch.norm(feature_maps, p=2, dim=1, keepdim=True)
    combinations['Std_Dev'] = torch.std(feature_maps, dim=1, keepdim=True)
    combinations['PCA'] = apply_pca_combination(feature_maps)
    return combinations

def apply_pca_combination(feature_maps, n_components=1):
    batch_size, num_channels, height, width = feature_maps.shape
    feature_flat = feature_maps.reshape(batch_size, num_channels, height * width)
    pca_results = []
    for b in range(batch_size):
        X = feature_flat[b].cpu().numpy()
        x_transposed = X.T
        pca = PCA(n_components=min(n_components, num_channels))
        x_pca = pca.fit_transform(x_transposed)
        x_pca_reshaped = x_pca.T.reshape(n_components, height, width)
        pca_tensor = torch.from_numpy(x_pca_reshaped).float()
        pca_results.append(pca_tensor)
    pca_combined = torch.stack(pca_results, dim=0)
    if n_components == 1:
        pca_combined = pca_combined
    return pca_combined

def apply_pca_multi_component(feature_maps, n_components=3):
    batch_size, num_channels, height, width = feature_maps.shape
    max_components = min(n_components, num_channels)
    pca_multi = apply_pca_combination(feature_maps, n_components=max_components)
    pca_combinations = {}
    pca_combinations['PCA_PC1'] = pca_multi[:, 0:1, :, :]
    if max_components >= 2:
        pca_combinations['PCA_PC2'] = pca_multi[:, 1:2, :, :]
        pca_combinations['PCA_PC1+PC2'] = (pca_multi[:, 0:1, :, :] * 0.7 +
                                          pca_multi[:, 1:2, :, :] * 0.3)
    if max_components >= 3:
        pca_combinations['PCA_PC3'] = pca_multi[:, 2:3, :, :]
        pca_combinations['PCA_PC1+PC2+PC3'] = (pca_multi[:, 0:1, :, :] * 0.5 +
                                              pca_multi[:, 1:2, :, :] * 0.3 +
                                              pca_multi[:, 2:3, :, :] * 0.2)
    return pca_combinations

def save_individual_images(
    combined_map, 
    original_image_path, 
    model_name, 
    pdp_stage, 
    combine_type, 
    output_dir
):
    base_image_name = os.path.splitext(os.path.basename(original_image_path))[0]
    filename = f"{base_image_name}_{model_name}_{pdp_stage}_{combine_type}.png"
    save_path = os.path.join(output_dir, "individual", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, combined_map, cmap='viridis')
    return save_path

def compare_models_combination_methods(model_features, output_dir):
    print("Creating separate images for each combination method...")
    if not model_features:
        print("No model features found!")
        return
    print(f"Found features for models: {list(model_features.keys())}")
    fr_pdp_organized = organize_fr_pdp_features(model_features)
    model_names = list(fr_pdp_organized.keys())
    pdp_stages = ['PDP1', 'PDP2', 'PDP3']
    if not model_names:
        print("No models found with FR-PDP blocks!")
        return
    print(f"Processing {len(model_names)} models: {model_names}")
    combination_methods = ['PCA', 'Mean', 'Max', 'RMS', 'Sum', 'Variance', 'Attention', 'L2_Norm', 'Std_Dev']
    for method_name in combination_methods:
        print(f"Creating visualization for {method_name} method...")
        fig, axes = plt.subplots(len(model_names), len(pdp_stages), figsize=(18, len(model_names) * 6))
        fig.suptitle(f'{method_name} Combination - All Models & FR-PDP Blocks',
                     fontsize=18, fontweight='bold')
        if len(model_names) == 1:
            axes = axes.reshape(1, -1)
        for model_idx, model_name in enumerate(model_names):
            for pdp_idx, pdp_stage in enumerate(pdp_stages):
                ax = axes[model_idx, pdp_idx]
                if pdp_stage in fr_pdp_organized[model_name]:
                    features = fr_pdp_organized[model_name][pdp_stage]
                    print(f"Processing {model_name}-{pdp_stage}: {features.shape}")
                    combinations = apply_all_combinations(features)
                    if method_name in combinations:
                        combined_map = combinations[method_name][0, 0].cpu().numpy()
                        combined_norm = (combined_map - combined_map.min()) / (combined_map.max() - combined_map.min() + 1e-8)
                        im = ax.imshow(combined_norm, cmap='viridis', aspect='auto')
                        save_individual_images(
                            combined_norm,
                            TARGET_IMAGE,
                            model_name,
                            pdp_stage,
                            method_name,
                            output_dir
                        )
                        mean_val = np.mean(combined_map)
                        std_val = np.std(combined_map)
                        max_val = np.max(combined_map)
                        min_val = np.min(combined_map)
                        ax.set_title(f'{model_name.upper()}\n{pdp_stage}\nμ={mean_val:.3f} σ={std_val:.3f}\n[{min_val:.2f}, {max_val:.2f}]',
                                   fontsize=10, fontweight='bold')
                        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
                        cbar.ax.tick_params(labelsize=8)
                    else:
                        ax.text(0.5, 0.5, f'{method_name}\nNot Available',
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=12, fontweight='bold')
                        ax.set_title(f'{model_name.upper()} - {pdp_stage}',
                                   fontsize=10, fontweight='bold')
                else:
                    print(f"{pdp_stage} not found in {model_name}")
                    ax.text(0.5, 0.5, f'{pdp_stage}\nNot Found',
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, fontweight='bold', color='red')
                    ax.set_title(f'{model_name.upper()} - {pdp_stage}',
                               fontsize=10, fontweight='bold')
                ax.axis('off')
        plt.tight_layout()
        method_filename = method_name.lower().replace('_', '-')
        output_path = os.path.join(output_dir, f'combination_{method_filename}_all_models.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {method_name} visualization: {output_path}")
    print(f"Generated {len(combination_methods)} separate combination method visualizations!")

def main():
    OUTPUT_DIR = "featuremap_compare_combined"
    TARGET_IMAGE = "datasets/StanfordDogs/Images/n02088364-beagle/n02088364_876.jpg"

    print(f"Device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = load_models()
    target_images = load_target_image(TARGET_IMAGE)
    fr_pdp_hook = FRPDPMultiModelHook()
    fr_pdp_hook.register_fr_pdp_hooks(models)
    model_features = fr_pdp_hook.extract_features(models, target_images)

    compare_models_combination_methods(model_features, OUTPUT_DIR)
    fr_pdp_hook.remove_all_hooks()
    print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
