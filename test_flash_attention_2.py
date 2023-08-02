from transformers import AutoConfig, AutoModel, EncodecModel, EncodecConfig, BarkCoarseConfig, BarkFineConfig, BarkSemanticConfig, BarkConfig, BarkFineModel, BarkCoarseModel, BarkSemanticModel, BarkModel, BarkProcessor, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.bark.generation_configuration_bark import BarkGenerationConfig
import torch
from huggingface_hub import HfApi, upload_folder, delete_folder, upload_file
import os
from transformers import set_seed
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write
from torch.autograd.graph import saved_tensors_hooks


from utils import generate_offload

api = HfApi()
set_seed(56379)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["SUNO_USE_SMALL_MODELS"] = "1"


from bark_modified import FlashAttention2BarkModel

################


        
def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return device

device = _grab_best_device()



processor = BarkProcessor.from_pretrained("ylacombe/bark-small")


sentences = [
    "I am not sure how to feel about it, but I've got a good feeling. It's clearly possible.",
    #"The force is strong here.",
    #"This is a test, I hope it will work [laughs]",
    #"Je suis une fleur sans épine, depuis que j'ai regardée dans tes yeux.",
]


tokenized_text = processor(sentences, "v2/en_speaker_6")


tokenized_text = tokenized_text.to(device)



SAMPLE_RATE = 24_000


HUB_PATH = "ylacombe/bark-small"

bark = FlashAttention2BarkModel.from_pretrained(HUB_PATH,)
                                 #torch_dtype=eval("torch.float16"))#, quantization_config=quantization_config)




bark = bark.to(device)

#bark.enable_cpu_offload(gpu_id = 0)



import time

# get the start time
st = time.time()

nb_loops = 10

for _ in range(nb_loops):
    with torch.inference_mode():
        output = bark.generate(**tokenized_text, do_sample = True, fine_temperature = 0.4, coarse_temperature = 0.8)
        
    for i in range(len(output)):
        write(f"test_audio_{i}.wav", SAMPLE_RATE, output[i, :].detach().cpu().numpy().squeeze().astype(np.float32))
    
    
# get the end time
et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time/nb_loops, 'seconds')