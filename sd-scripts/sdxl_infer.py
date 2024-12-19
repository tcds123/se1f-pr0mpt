import os

from diffusers import StableDiffusionXLPipeline
import torch
import json
import random
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_path = "/data/team/zongwx1/llm_models/stable-diffusion-xl-base-1.0/Juggernaut_X_RunDiffusion.safetensors"
model_path = "/data/team/zongwx1/llm_models/stable-diffusion-xl-base-1.0/AnythingXL_xl.safetensors"
model_path = "/data/team/zongwx1/llm_models/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
# model_path = "./Juggernaut_X_RunDiffusion.safetensors"

pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    # config=config,
    original_config='./sd_xl_base.yaml',
    torch_dtype=torch.float16,
    local_files_only=True,
    use_safetensors=True,
    add_watermarker=False,
).to(DEVICE)


def set_random_seed(seed):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def image_generation(prompt, prompt_2=None, save_path=None, height=512, width=512, seed=None,
                     negative_prompt=None, negative_prompt_2=None):
    if seed:
        set_random_seed(seed)
    image = pipe(prompt=prompt,
                 prompt_2=prompt_2,
                 num_inference_steps=25,
                 guidance_scale=9,
                 height=height,
                 width=width,
                 negative_prompt=negative_prompt,
                 negative_prompt_2=negative_prompt_2).images[0]
    if save_path:
        image.save(save_path)
        return
    return image


def ori_image_generation(dataset_path, save_path, seed):
    with open('./dataset/laion/params.json', "r") as f:
        params = json.load(f)
    txt_path_list = os.listdir(dataset_path + 'text')
    for txt_path in txt_path_list:
        index = txt_path.split('.')[0]
        if index == '':
            continue
        height = params[index]['height'] - params[index]['height'] % 8
        width = params[index]['width'] - params[index]['width'] % 8
        try:
            with open(dataset_path + 'text/' + txt_path, "r", encoding='utf-8') as f:
                prompt = f.read()
        except Exception as e:
            print(e)
            continue
        print(prompt)
        image_generation(prompt, save_path=save_path + index + '.png', height=height, width=width, seed=seed)
    print("finish")


def optimizied_image_generation(dataset_path, save_path, seed):
    with open('./dataset/laion/params.json', "r") as f:
        params = json.load(f)
    txt_path_list = os.listdir(dataset_path + 'text')
    extra_prompt_path = dataset_path + 'extra_prompt/'
    for txt_path in txt_path_list:
        index = txt_path.split('.')[0]
        if index == '':
            continue
        height = params[index]['height'] - params[index]['height'] % 8
        width = params[index]['width'] - params[index]['width'] % 8
        with open(dataset_path + 'text/' + txt_path, "r", encoding='utf-8') as f:
            prompt = f.read()
        try:
            with open(extra_prompt_path + index + '_1.txt', "r") as f:
                prompt1 = f.read()
            with open(extra_prompt_path + index + '_2.txt', "r") as f:
                prompt2 = f.read()
            with open(extra_prompt_path + 'neg_' + index + '_1.txt', "r") as f:
                neg_prompt1 = f.read()
            with open(extra_prompt_path + 'neg_' + index + '_2.txt', "r") as f:
                neg_prompt2 = f.read()
        except Exception:
            print("no extra prompt")
            continue
        prompt = prompt + prompt1 + prompt2
        neg_prompt = neg_prompt1 + neg_prompt2
        print(prompt)
        print(neg_prompt)
        image_generation(prompt, save_path=save_path + index + '.png', height=height, width=width, seed=seed, negative_prompt=neg_prompt)
    print("finish")


def ablation_random_neg_prompt(dataset_path, save_path, seed):
    with open(dataset_path + 'params.json', "r") as f:
        params = json.load(f)
    txt_path_list = os.listdir(dataset_path + 'text')
    extra_prompt_path = dataset_path + 'extra_prompt/'
    for txt_path in txt_path_list:
        index = txt_path.split('.')[0]
        if index == '':
            continue
        height = params[index]['height'] - params[index]['height'] % 8
        width = params[index]['width'] - params[index]['width'] % 8
        with open(dataset_path + 'text/' + txt_path, "r") as f:
            prompt = f.read()
        try:
            with open(extra_prompt_path + index + '_1.txt', "r") as f:
                prompt1 = f.read()
            with open(extra_prompt_path + index + '_2.txt', "r") as f:
                prompt2 = f.read()
            with open(extra_prompt_path + 'neg_' + index + '_1.txt', "r") as f:
                neg_prompt1 = f.read()
            with open(extra_prompt_path + 'neg_' + index + '_2.txt', "r") as f:
                neg_prompt2 = f.read()
        except Exception:
            print("no extra prompt")
            continue
        prompt = prompt + prompt1 + prompt2
        # prompt = prompt1 + prompt2
        prompt = prompt

        # neg_prompt1 = neg_prompt2 = ""
        # neg_prompt1 = neg_prompt_random_init(length=10)
        # neg_prompt2 = neg_prompt1
        neg_prompt = neg_prompt1 + neg_prompt2
        print(prompt)
        print(neg_prompt)
        image_generation(prompt, save_path=save_path + index + '.png', height=height, width=width, seed=seed, negative_prompt=neg_prompt)
    print("finish")


def main_pre_gen():
    for seed in range(1, 6):
        txt_dir = "./dataset/laion/text/"
        extra_prompt_path = f"./dataset/laion/extra_prompt/{seed}/"
        ori_save_path = f"./dataset/laion/ori_image/{seed}/"
        opt_save_path = f"./dataset/laion/optimized_image/{seed}/"
        if not os.path.exists(ori_save_path):
            os.mkdir(ori_save_path)
        if not os.path.exists(opt_save_path):
            os.mkdir(opt_save_path)
        with open('./dataset/laion/params.json', "r") as f:
            params = json.load(f)
        txt_path_list = os.listdir(txt_dir)
        count = 0
        for txt_path in txt_path_list:
            index = txt_path.split('.')[0]
            if index == '':
                continue
            height = params[index]['height'] - params[index]['height'] % 8
            width = params[index]['width'] - params[index]['width'] % 8
            with open(txt_dir + txt_path, "r", encoding='utf-8') as f:
                prompt = f.read()
            try:
                with open(extra_prompt_path + index + '_1.txt', "r") as f:
                    prompt1 = f.read()
                with open(extra_prompt_path + index + '_2.txt', "r") as f:
                    prompt2 = f.read()
                with open(extra_prompt_path + 'neg_' + index + '_1.txt', "r") as f:
                    neg_prompt1 = f.read()
                with open(extra_prompt_path + 'neg_' + index + '_2.txt', "r") as f:
                    neg_prompt2 = f.read()
            except Exception:
                print("no extra prompt")
                continue
            image_generation(prompt, save_path=ori_save_path + index + '.png', height=height, width=width, seed=seed)
            prompt = prompt + prompt1 + prompt2
            neg_prompt = neg_prompt1 + neg_prompt2
            print(prompt)
            print(neg_prompt)
            image_generation(prompt, save_path=opt_save_path + index + '.png', height=height, width=width, seed=seed,
                             negative_prompt=neg_prompt)
            count += 1
            print(f"seed: {seed}, {count} / {len(txt_path_list)}")
        print("finish")


def main_ori_generation(seed):
    dataset_path = "./dataset/laion/"
    save_path = dataset_path + "ori_image/"
    ori_image_generation(dataset_path, save_path, seed=seed)


def main_optimize_generation(seed):
    dataset_path = "./dataset/laion/"
    save_path = dataset_path + "optimized_image/"
    optimizied_image_generation(dataset_path, save_path, seed=seed)


def main_ablation(seed):
    dataset_path = "./dataset/laion/"
    save_path = dataset_path + "optimized_image/"
    ablation_random_neg_prompt(dataset_path, save_path, seed=seed)


def main_apio_generation(seed):
    dataset_path = "./dataset/laion/apio/"
    save_path = dataset_path + "image/"
    ori_image_generation(dataset_path, save_path, seed=seed)


def main_sdxl_base_generation(seed):
    dataset_path = "./dataset/laion/"
    save_path = dataset_path + "sd_xl_base_1.0/ori_image/"
    ori_image_generation(dataset_path, save_path, seed=seed)


def neg_prompt_random_init(length):
    sp_token = [49406, 49407]
    tokenizer = pipe.tokenizer
    vocab_size = tokenizer.vocab_size
    neg_prompt = []
    for i in range(length):
        neg_token = random.randint(0, vocab_size - 1)
        while neg_token in sp_token:
            neg_token = random.randint(0, vocab_size - 1)
        neg_prompt.append(neg_token)
    neg_prompt = tokenizer.decode(neg_prompt)
    return neg_prompt



if __name__ == "__main__":
    # prompt = "The painting captures a serene moment in nature. At the center, a calm lake reflects the sky, its surface rippled only by the gentlest of breezes. The sky above is a brilliant mix of blues and whites, with fluffy clouds drifting leisurely across. On the banks of the lake, tall trees stand gracefully, their leaves rustling in the wind. In the foreground, an old man sits on a rock, seemingly lost in deep thought or meditation. The soft light of the setting sun bathes the entire scene in a warm glow, creating a sense of peace and tranquility. The colors are muted yet vibrant, and the details are captured with precision, giving the painting a sense of realism while still retaining a dreamlike quality."
    # prompt1 = "The painting captures a serene moment in nature. At the center, a calm lake reflects the sky, its surface rippled only by the gentlest of breezes. The sky above is a brilliant mix of blues and whites, with fluffy clouds drifting leisurely across. On the banks of the lake, tall trees stand gracefully, their leaves rustling in the wind. "
    # prompt_2 = "In the foreground, an old man sits on a rock, seemingly lost in deep thought or meditation. The soft light of the setting sun bathes the entire scene in a warm glow, creating a sense of peace and tranquility. The colors are muted yet vibrant, and the details are captured with precision, giving the painting a sense of realism while still retaining a dreamlike quality."
    # image_generation(prompt1, prompt_2, save_path="catdog.png")

    # seed = 1
    # set_random_seed(seed)
    # main_ori_generation(seed)
    # main_optimize_generation(seed)

    # main_ablation(seed)
    # main_apio_generation(seed)
    # main_sdxl_base_generation(seed)
    main_pre_gen()
