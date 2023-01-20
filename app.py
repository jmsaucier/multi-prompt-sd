import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import base64
from io import BytesIO
import os
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from contextlib import contextmanager, nullcontext
from tqdm import tqdm, trange
from PIL import Image
from einops import rearrange
import numpy as np

def load_model_from_config(config, ckpt="model.ckpt", verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    #     print("missing keys:")
    #     print(m)
    # if len(u) > 0 and verbose:
    #     print("unexpected keys:")
    #     print(u)

    model.cuda()
    model.eval()
    return model


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    # HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    
    # this will substitute the default PNDM scheduler for K-LMS  
    # lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    # model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
    #                                                 scheduler=lms, 
    #                                                 use_auth_token=HF_AUTH_TOKEN
    #                                                 ).to("cuda")
    
    config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    
    [weight1, weight2] = model_inputs.get('weights', [None, None])

    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)
    
    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    
    averagedWeights = None

    if weight1 != None and weight2 != None: 
        print('do stuff with weights 1 and 2')
    if prompt == None:
        return {'message': "No prompt provided"}
    
    data = [[prompt]]
    sampler = DDIMSampler(model)
    Anon = lambda **kwargs: type("Object", (), kwargs)
    opt = Anon(
        C = 4,
        W = width,
        H = height,
        f = 8,
        scale = 7.5,
        ddim_steps = 25,
        n_samples = 1,
        ddim_eta = 0.0,
    )

    batch_size = 1
    image = None
    encoding = None
    with torch.no_grad():
        with autocast("cuda"):
            with model.ema_scope():
                all_samples = list()
                
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    conditioning = model.get_learned_conditioning(prompts)
                    encoding = conditioning.cpu().numpy().tolist()
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=conditioning,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=None)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_image = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    for x_sample in x_image:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            image = Image.fromarray(x_sample.astype(np.uint8))
                            


    
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {
        # 'image_base64': image_base64, 
        'encoding': encoding
        }