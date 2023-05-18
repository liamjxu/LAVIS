import torch
from PIL import Image
from IPython.display import display
from lavis.models import load_model_and_preprocess


# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=torch.device("cuda"))


# load sample image
def image_gen(image_path, prompt="What is unusual about this image?", device=torch.device("cuda")):
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    return model.generate({"image": image, "prompt": prompt})


image_gen('chart.png')