#!/usr/bin/env python3
import sys
import json
import re
import pathlib
from pngmeta import PngMeta
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers import DiffusionPipeline
from diffusers import StableDiffusion3Pipeline
import argparse

models = {
    "sd15": {"path": "runwayml/stable-diffusion-v1-5",
             "desc": "Stable Diffusion 1.5 standard",
             "pipeline": "sdp", "pipeline_options": {}, "img_gen_option": {}},
    "sd14": {"path": "CompVis/stable-diffusion-v1-4",
             "desc": "Stable Diffusion 1.4 standard",
             "pipeline": "sdp", "pipeline_options": {}, "img_gen_option": {}},
    "sd21": {"path": "stabilityai/stable-diffusion-2-1",
             "desc": "Stable Diffusion 2.1 standard", "pipeline": "sdp",
             "pipeline_options": {}, "img_gen_option": {}},
    "sd21b":{"path":  "stabilityai/stable-diffusion-2-1-base",
            "desc": "Stable Diffusion 2.1 base",
            "pipeline": "sdp", "pipeline_options": {}, "img_gen_option": {}},
    "sd35t":{"path": "stabilityai/stable-diffusion-3.5-large-turbo",
            "desc": "Stable Diffusion 3.5 turbo",
            "pipeline": "sd3p", "pipeline_options": {}, "img_gen_option": {}},
    "sd35m":{"path": "stabilityai/stable-diffusion-3.5-medium",
            "desc": "Stable Diffusion 3.5 medium",
            "pipeline": "sd3p", "pipeline_options": {}, "img_gen_option": {}},
    "sd3m": {"path": "ckpt/stable-diffusion-3-medium-diffusers",
            "desc": "Stable Diffusion 3.0 medium",
            "pipeline": "sd", "pipeline_options": {}, "img_gen_option": {}},
    "ds8":  {"path": "Lykon/DreamShaper-8",
            "desc": "Photorealistic model based on Stable Diffusion 1.5 standard",
            "pipeline": "sdp", "pipeline_options": {}, "img_gen_option": {}}, #photorealistic
    "rv":   {"path": "SG161222/Realistic_Vision_V1.4",
            "desc": "Photorealistic based on Stable Diffusion 1.4",
            "pipeline": "sdp", "pipeline_options": {}, "img_gen_option": {}},
    "cf":   {"path": "gsdf/Counterfeit-V2.5",
            "desc": "Manga and Anime tuned model",
            "pipeline": "sdp", "pipeline_options": {}, "img_gen_option": {}},
    "mm":   {"path": "Meina/MeinaMix_V11",
            "desc": "MeniaMix base model for anime and manga",
            "pipeline": "sdp", "pipeline_options": {}, "img_gen_option": {}},
    "ill":  {"path": "OnomaAIResearch/Illustrious-xl-early-release-v0",
            "desc": "Based on Stable Diffusion XL fine-tuned on Danbooru2023 Dataset, uncensored model",
            "pipeline": "sdXLp",
            "pipeline_options": {},
            "img_gen_option": {"original_size": (512,512), "target_size": (1024,1024), "crops_coords_top_left": (0,0)}},
    "yoda": {"path": "yodayo-ai/kivotos-xl-2.0",
            "desc": "Kivotos XL V2.0, Open-source model built upon Animagine XL V3, a specialized for generating high-quality anime-style artwork.",
            "pipeline": "sdXLp",
            "pipeline_options": {"use_safetensors": True, "custom_pipeline": "lpw_stable_diffusion_xl", "torch_dtype": torch.float16, "add_watermarker": False, "variant": "fp16"},
            "img_gen_option":  {"original_size": (512,512), "target_size": (1024,1024), "crops_coords_top_left": (0,0)}}
            # prompt = "1girl, kazusa \(blue archive\), blue archive, solo, upper body, v, smile, looking at viewer, outdoors, night, masterpiece, best quality, very aesthetic, absurdres"
            # negative_prompt = "nsfw, (low quality, worst quality:1.2), very displeasing, 3d, watermark, signature, ugly, poorly drawn"
    }

summary = {}


def parseArguments():
    parser = argparse.ArgumentParser(
        prog='Stable Diffusion Image Generator',
        description='Generate an image with Stable Diffusion')

    parser.add_argument('-p', '--prompt', type=str, default="a drowing an astronaut riding a horse")
    parser.add_argument('-np', '--negative-prompt', type=str, default="bad eyes, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits")
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--steps', type=int, default=10)
    parser.add_argument('--seed', type=int)
    parser.add_argument('-t', '--temperature', type=float, default=7.5)
    parser.add_argument('-o', '--out-prefix', type=str)
    parser.add_argument('-m', '--model', type=str, default="def")
    parser.add_argument('-sfw', type=bool, default=False)

    return parser.parse_args()

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def getPipeline(model, sfw):

    if model == "def":
        model = "sd15"

    modelInfo = models[model]
    model_path = modelInfo["path"]

    summary["model"] = modelInfo
    summary["sfw"] = sfw

    # safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
    pipe_options = modelInfo["pipeline_options"]
    if sfw == False:
        print("preparing safety checker")
        pipe_options["safety_checker"] = None
    pipe = None
    match modelInfo["pipeline"]:
        case "sdXLp":
            del pipe_options["safety_checker"]
            #print(f'Current pipiline options: {pipe_options}')
            pipe = StableDiffusionXLPipeline.from_pretrained(model_path, **pipe_options)
        case "sdp":
            pipe = StableDiffusionPipeline.from_pretrained(model_path, **pipe_options)
        case "sd3p":
            del pipe_options["safety_checker"]
            #print(f'Current pipiline options: {pipe_options}')
            pipe = StableDiffusion3Pipeline.from_pretrained(model_path, **pipe_options)
        case "sd":
            pipe = DiffusionPipeline.from_pretrained(model_path, **pipe_options)
        case _:
            pipe = StableDiffusionPipeline.from_pretrained(model_path, **pipe_options)
    print(f'Current pipiline options: {pipe_options}')

    return pipe

def generateImage(prompt, neg_prompt, model, sfw, device, iterate=1, temp=7.5, seed=-1, steps=10):

    summary["device"] = device
    summary["prompt"] = prompt
    summary["negative prompt"] = neg_prompt
    summary["steps"] = steps
    summary["temperature"] = temp

    pipe = getPipeline(model, sfw)
    pipe = pipe.to(device)

    img_gen_option = models[model]["img_gen_option"]
    print(f'Current image generation options: {img_gen_option}')

    if seed:
        print(f'manual seed set to: {seed}')
    else:
        seed = torch.seed()
    print(f'seed is set to: {seed}')
    summary["seed"] = seed
    generator = torch.Generator(device).manual_seed(seed)

    prompts = [f'{prompt}']*iterate

    # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
    #here we need to insert the options stored in img_gen_option
    images = pipe(prompts, negative_prompt=[neg_prompt], num_inference_steps=steps, guidance_scale=temp, generator=generator, **img_gen_option).images

    return images

def getFilesList(prefix):
    directory_path = "./"
    directory = pathlib.Path(directory_path)

    pattern = prefix + "*.png"
    #print(pattern)

    found_files = []
    if not directory.is_dir():
        print(f"ERROR: The directory '{directory_path}' does not exist.")
    else:
        found_files = list(directory.glob(pattern))
        #print(f'found files: {found_files}')
    return found_files

def initCounter(prefix):
    #print("------ initCounter ------")


    file_names = getFilesList(prefix)

    regex_pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)\.png$")

    max_index = -1
    for fname in file_names:
        name = fname.name
        match = regex_pattern.match(name)
        if match:
            try:
                # Estrai il gruppo catturato (l'indice) e convertilo in intero
                current_index = int(match.group(1))

                # Aggiorna l'indice massimo
                if current_index > max_index:
                    max_index = current_index
            except ValueError:
                # Questo è improbabile data la regex (\d+), ma è una buona pratica
                print(f"Warning: Cannot convert index for file: {name}")

    #print("------------")
    return max_index + 1

def addMetadata(image_name, meta_data):
    meta = PngMeta(image_name)
    meta['Comment'] = meta_data
    meta.save()

def storeImage(image, img_name):
    print(f'Storing image as \"{img_name}\"')
    image.save(img_name)
    meta_data = json.dumps(summary, indent=4)
    addMetadata(img_name, meta_data)

    try:
        with open(img_name+".txt", 'w', encoding='utf-8') as f:
            f.write(meta_data)

        print(f'Summary exported to: {img_name+".txt"}')

    except Exception as e:
        print(f"❌ Errore durante il salvataggio: {e}")

def storeImages(images, prefix):
    # Now to display an image you can either save it such as:
    counter=initCounter(prefix)
    files = []
    for image in images:
        img_name = f"{prefix}-{counter}.png"
        files.append(img_name)
        storeImage(image, img_name)
        counter += 1
    #print(files)
    summary["images"] = files
    #print(summary)

def printSummary():
    #print(summary)
    #for item in summary:
        #print(f'{item}: {summary[item]}')
    summary_json = json.dumps(summary, indent=4)
    #print(summary_json)

def storeImagesToGrind():

    grid = image_grid(images, rows=1, cols=3)
    return


def main():
    args = parseArguments()

    file_prefix = "genimage"
    if args.out_prefix == None:
        if args.model != "def":
            file_prefix == args.model
            #print(f'file_prefix: {file_prefix}')
    else:
        file_prefix = args.out_prefix
        #print(f'file_prefix: {file_prefix} !!!')

    device="cpu"
    if torch.cuda.is_available():
        device=torch.device("cuda")
    elif torch.backends.mps.is_available():
        device=torch.device("mps")
    print(f'device is: {device}')

    generated_images = generateImage(args.prompt, args.negative_prompt, args.model, args.sfw, device, steps=args.steps, temp=args.temperature, seed=args.seed, iterate=args.batch)
    storeImages(generated_images, file_prefix)
    printSummary()

    print("Image generation complete!")

if __name__ == '__main__':
    main()


