# Funciones base para la evaluación manual y automática de Textual Inversión
# LoRA y DreamBooth
import torch
from torch import autocast
import os
from PIL import Image
from functools import partial
from torchmetrics.functional.multimodal import clip_score
import numpy as np

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
def calculate_clip_score(images_np, prompts):
    """
    Calcula el CLIP Score promedio entre imágenes y prompts.
    """
    images_int = (images_np * 255).astype("uint8")  # a uint8
    images_tensor = torch.from_numpy(images_int).permute(0, 3, 1, 2)  # (N, C, H, W)
    scores = clip_score_fn(images_tensor, prompts)
    return scores.detach().cpu().numpy()


def generate_images(prompts,
                    pipe,
                    names_export,
                    negative_prompts = [],
                    n_repeats=3,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    num_rows=1,
                    num_samples = 3
                    ):
    all_images = []
    prompts_repeated = []
    all_clips = []
    for i in range(0,len(prompts)):
        prompt = prompts[i]
        print("Prompt: " + prompt)

        negative_prompt = negative_prompts[i]
        print("Negative prompt: " + negative_prompt)

        name_export = names_export[i]

        # Generación de imágenes
        all_images_prompt = []
        # Generador de seeds reproducibles (opcional)
        g_cuda = torch.Generator(device="cuda").manual_seed(42)

        for _ in range(num_rows):
            with autocast("cuda"), torch.inference_mode():
                images = pipe(
                    [prompt] * num_samples,
                    height=height,
                    width=width,
                    negative_prompt=[negative_prompt] * num_samples,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1,
                    generator=g_cuda
                ).images

                all_images_prompt.extend(images)
                all_images.extend(images)

        # Mostrar grid. las tres imágenes por prompt
        grid = image_grid(all_images_prompt, num_rows, num_samples)
        display(grid)

        # Guarda el trio de imagenes en local
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)

        grid_filename = f"{output_dir}/{name_export}.png"
        grid.save(grid_filename)
        print(f"Grid guardado en: {grid_filename}")

        # Calcula el CLIP
        clip_group = calculate_clip_score(np.array(all_images_prompt),
         [prompt] * n_repeats)
        all_clips.extend([clip_group.item()])
        print(clip_group)

    prompts_repeated = []
    for prompt in prompts:
        prompts_repeated.extend([prompt] * n_repeats)

    images_np = np.array(all_images)

    # calculo el clip para el conjunto de prompts del criterio
    scores = calculate_clip_score(images_np, prompts_repeated)
    return(scores, all_clips)
