# Funciones base para la evaluación manual y automática de Textual Inversión
# LoRA y DreamBooth

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


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
                    negative_prompts = [],
                    n_repeats=3,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                    ):
    all_images = []
    prompts_repeated = []
    all_clips = []
    for i in range(0,len(prompts)):
        prompt = prompts[i]
        print("Prompt: " + prompt)

        negative_prompt = negative_prompts[i]
        print("Negative prompt: " + negative_prompt)

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