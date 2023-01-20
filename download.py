import urllib.request

def download_model():
    # # do a dry run of loading the huggingface model, which will download weights at build time
    # #Set auth token which is required to download stable diffusion model weights
    # HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    # lms = LMSDiscreteScheduler(
    #     beta_start=0.00085, 
    #     beta_end=0.012, 
    #     beta_schedule="scaled_linear"
    # )
    
    modelUrl = 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt'

    urllib.request.urlretrieve(modelUrl, 'model.ckpt')

    # model = StableDiffusionPipeline.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4", 
    #     scheduler=lms,
    #     use_auth_token=HF_AUTH_TOKEN
    # )

if __name__ == "__main__":
    download_model()