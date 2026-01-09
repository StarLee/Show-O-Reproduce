# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image,ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F


from PIL import Image, ImageDraw, ImageFont

def add_caption_below(img: Image.Image, caption_text, output_path):
    """
    在图片下方添加文字说明并保存。
    如果文字过长，会自动换行以适应图片宽度。
    """
    w, h = img.size

    # 1. 设置字体
    # 策略：图片越宽字体越大，但设定一个最小和最大值防止过大或过小
    try:
        font_size = int(w / 25)  # 稍微调小一点比例，容纳更多字
        font_size = max(20, min(font_size, 60)) # 限制字号在 20-60 之间
        # Windows使用 msyh.ttc, Linux/Mac 请替换为合适的字体路径
        font = ImageFont.truetype("msyh.ttc", font_size) 
    except OSError:
        font = ImageFont.load_default()

    # 2. 自动换行逻辑 (核心修改部分)
    draw_temp = ImageDraw.Draw(img)
    lines = []
    current_line = ""
    
    # 设定文字允许的最大宽度 (图片宽度 - 左右边距)
    side_padding = 20 
    max_text_width = w - (side_padding * 2)
    
    # 逐字检查宽度
    for char in caption_text:
        # 检查如果加上这个字符，是否会超宽
        if draw_temp.textlength(current_line + char, font=font) <= max_text_width:
            current_line += char
        else:
            # 如果超宽，把当前行加入列表，并重新开始新的一行
            lines.append(current_line)
            current_line = char
    # 把最后剩下的内容加入
    if current_line:
        lines.append(current_line)

    # 3. 计算所有文字所需的总高度
    # 获取单行文字的高度 (假设所有行高度一致)
    # textbbox 返回 (left, top, right, bottom)
    bbox = draw_temp.textbbox((0, 0), "测试Text", font=font)
    line_height = bbox[3] - bbox[1]
    # 行间距
    line_spacing = 10 
    
    # 文字区域总高度 = (行数 * 单行高) + ((行数-1) * 行间距)
    total_text_height = len(lines) * line_height + (len(lines) - 1) * line_spacing

    # 4. 创建新画布
    # 顶部边距 + 底部边距
    vertical_padding = 30
    new_height = h + total_text_height + (vertical_padding * 2)
    
    new_img = Image.new("RGB", (w, new_height), "white")

    # 5. 粘贴原图
    new_img.paste(img, (0, 0))

    # 6. 逐行写入文字
    draw = ImageDraw.Draw(new_img)
    
    # 起始 Y 坐标 (原图高度 + 顶部padding)
    current_y = h + vertical_padding
    
    for line in lines:
        # 计算当前行的宽度，以便居中
        line_width = draw.textlength(line, font=font)
        text_x = (w - line_width) / 2
        
        draw.text((text_x, current_y), line, font=font, fill="black")
        
        # Y 坐标下移
        current_y += line_height + line_spacing

    # 7. 保存
    new_img.save(output_path)
    print(f"已保存到: {output_path} (共 {len(lines)} 行文字)")





def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name + '_t2i' + f'_{config.mode}',
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model.eval()

    mask_token_id = model.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = config.batch_size
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps
    # load from users passed arguments

    if config.mode == 'inpainting':
        import os
        if not os.path.exists(f"./results/{config.mode}"):
                os.makedirs(f"./results/{config.mode}")
        prompt = [config.prompt] * config.batch_size
        inpainting_image = Image.open(config.image_path).convert("RGB")
        inpainting_mask = Image.open(config.inpainting_mask_path).convert("L")

        inpainting_image = image_transform(inpainting_image, resolution=config.dataset.params.resolution).to(device)
        inpainting_mask = image_transform(inpainting_mask, resolution=config.dataset.params.resolution, normalize=False)

        # record original image and inpainting mask
        images = torch.clamp(
            (torch.stack([inpainting_image, inpainting_mask.repeat(3, 1, 1).to(device)], dim=0) + 1.0) / 2.0,
            min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        labels = ['original image', 'inpainting mask']
        for i,image in enumerate(pil_images):
            add_caption_below(image, f"label: {labels[i]}", f"./results/{config.mode}/{config.mode}_input_{labels[i]}.png")
        wandb_images = [wandb.Image(image, caption=labels[i]) for i, image in enumerate(pil_images)]

        inpainting_image = inpainting_image.unsqueeze(0).repeat(config.training.batch_size, 1, 1, 1)

        inpainting_mask = inpainting_mask.unsqueeze(0).to(device)
        inpainting_mask = F.interpolate(inpainting_mask, size=config.dataset.params.resolution // 16, mode='bicubic')
        inpainting_mask = inpainting_mask.repeat(config.training.batch_size, 1, 1, 1)

        inpainting_mask[inpainting_mask < 0.5] = 0
        inpainting_mask[inpainting_mask >= 0.5] = 1

        inpainting_mask = inpainting_mask.reshape(config.training.batch_size, -1)
        inpainting_mask = inpainting_mask.to(torch.bool)

        inpainting_image_tokens = vq_model.get_code(inpainting_image) + len(uni_prompting.text_tokenizer)
        inpainting_image_tokens[inpainting_mask] = mask_token_id

        input_ids, _ = uni_prompting((prompt, inpainting_image_tokens), 't2i_gen')

        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(prompt), inpainting_image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
            uncond_input_ids = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

        with torch.no_grad():
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.showo.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
        # import ipdb
        # ipdb.set_trace()
        
        for i,img in  enumerate(pil_images):
            #img.save(f"./results/{config.mode}/{config.mode}_{i}.png")
            add_caption_below(img, f"prompt: {prompt[i]}", f"./results/{config.mode}/{config.mode}_{i}.png")
        wandb_images.extend([wandb.Image(image, caption=prompt[i]) for i, image in enumerate(pil_images)])
        wandb.log({"generated_images": wandb_images}, step=0)

    elif config.mode == 'extrapolation':
        if not os.path.exists(f"./results/{config.mode}"):
                os.makedirs(f"./results/{config.mode}")
        prompt = [p for p in config.prompt.split(" *** ") if len(p) != 0]
        extra_direction = [d for d in config.extra_direction.split(" *** ") if len(d) != 0]
        print(prompt, extra_direction)
        W = config.dataset.params.resolution // 16
        for id, (prt, direction) in enumerate(zip(prompt, extra_direction)):
            prt = [prt] * config.training.batch_size
            if id == 0:
                extrapolation_image = Image.open(config.image_path).convert("RGB")
                add_caption_below(extrapolation_image, "original image", f"./results/{config.mode}/{config.mode}_original_image.png")
                extrapolation_image = image_transform(extrapolation_image,
                                                      resolution=config.dataset.params.resolution).to(device)

                B, _, _ = extrapolation_image.shape
                extrapolation_image = extrapolation_image.unsqueeze(0)
                extrapolation_image_tokens = vq_model.get_code(extrapolation_image) + len(uni_prompting.text_tokenizer)
                extrapolation_image_tokens = extrapolation_image_tokens.reshape(1,
                                                                                config.dataset.params.resolution // 16,
                                                                                config.dataset.params.resolution // 16)
                extrapolation_image_tokens = extrapolation_image_tokens.repeat(config.training.batch_size, 1, 1)
            else:
                extrapolation_image_tokens = gen_token_ids + len(uni_prompting.text_tokenizer)

            image_left_part = extrapolation_image_tokens[:, :, :-(W//2-config.offset)] - len(uni_prompting.text_tokenizer)
            image_right_part = extrapolation_image_tokens[:, :, W//2-config.offset:] - len(uni_prompting.text_tokenizer)
            image_up_part = extrapolation_image_tokens[:, :-(W//2-config.offset), :] - len(uni_prompting.text_tokenizer)
            image_down_part = extrapolation_image_tokens[:, W//2-config.offset:, :] - len(uni_prompting.text_tokenizer)

            if direction in ['left', 'right']:
                extrapolation_mask = torch.zeros((config.training.batch_size,
                                                  config.dataset.params.resolution // 16,
                                                  config.dataset.params.resolution // 16 // 2 + config.offset),
                                                 dtype=torch.int64, device=device) + mask_token_id
            else:
                extrapolation_mask = torch.zeros((config.training.batch_size,
                                                  config.dataset.params.resolution // 16 // 2 + config.offset,
                                                  config.dataset.params.resolution // 16),
                                                 dtype=torch.int64, device=device) + mask_token_id

            if direction == 'left':
                extrapolation_image_tokens = torch.cat(
                    [extrapolation_mask, extrapolation_image_tokens[:, :, :W//2-config.offset]], dim=-1)
            elif direction == 'right':
                extrapolation_image_tokens = torch.cat(
                    [extrapolation_image_tokens[:, :, -(W//2-config.offset):], extrapolation_mask], dim=-1)
            elif direction == 'up':
                extrapolation_image_tokens = torch.cat(
                    [extrapolation_mask, extrapolation_image_tokens[:, :W // 2 - config.offset, :]], dim=-2)
            else:
                extrapolation_image_tokens = torch.cat(
                    [extrapolation_image_tokens[:, -(W // 2 - config.offset):, :], extrapolation_mask], dim=-2)

            extrapolation_image_tokens = extrapolation_image_tokens.reshape(config.training.batch_size, -1)

            input_ids, _ = uni_prompting((prt, extrapolation_image_tokens), 't2i_gen')

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(prt), extrapolation_image_tokens), 't2i_gen')
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                uncond_input_ids = None

            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

            with torch.no_grad():
                gen_token_ids = model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            gen_token_ids = gen_token_ids.reshape(config.training.batch_size,
                                                  config.dataset.params.resolution // 16,
                                                  config.dataset.params.resolution // 16)
            if direction == 'left':
                gen_token_ids = torch.cat([gen_token_ids, image_right_part], dim=-1)
            elif direction == 'right':
                gen_token_ids = torch.cat([image_left_part, gen_token_ids], dim=-1)
            elif direction == 'up':
                gen_token_ids = torch.cat([gen_token_ids, image_down_part], dim=-2)
            else:
                gen_token_ids = torch.cat([image_left_part, gen_token_ids], dim=-2)

        _, h, w = gen_token_ids.shape
        gen_token_ids = gen_token_ids.reshape(config.training.batch_size, -1)
        images = vq_model.decode_code(gen_token_ids, shape=(h, w))

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
       
       
        for i,img in  enumerate(pil_images):
                #img.save(f"./results/{config.mode}/{config.mode}_{i}.png")
            add_caption_below(img, f"prompt: {' '.join(prompt)}", f"./results/{config.mode}/{config.mode}_{i}.png")
        wandb_images = [wandb.Image(image, caption=' '.join(prompt)) for i, image in enumerate(pil_images)]
        wandb.log({"generated_images": wandb_images}, step=0)

    elif config.mode == 't2i':
        with open(config.dataset.params.validation_prompts_file, "r") as f:
            validation_prompts = f.read().splitlines()

        for step in tqdm(range(0, len(validation_prompts), config.training.batch_size)):
            prompts = validation_prompts[step:step + config.training.batch_size]

            image_tokens = torch.ones((len(prompts), config.model.showo.num_vq_tokens),
                                      dtype=torch.long, device=device) * mask_token_id

            input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                uncond_input_ids = None

            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

            with torch.no_grad():
                gen_token_ids = model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]

            wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
            import os
            if not os.path.exists(f"./results/{config.mode}"):
                os.makedirs(f"./results/{config.mode}")
            for i,img in  enumerate(pil_images):
                add_caption_below(img, f"prompt:{prompts[i]}", f"./results/{config.mode}/{config.mode}_{step}_{i}.png")
                
                
                
            wandb.log({"generated_images": wandb_images}, step=step)
    wandb.finish()