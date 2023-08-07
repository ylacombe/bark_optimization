import os
import argparse

import torch
import transformers
from transformers import (
    BarkModel,
    BarkProcessor,
)

from bark_modified import FlashAttentionBarkModel, TestBarkModel, BarkVocosFine, BarkVocosCoarse, BarkVocosCoarseStream

from bark.api import generate_audio
from bark.generation import preload_models, SAMPLE_RATE
from vocos import Vocos

import torch
from transformers import set_seed
from datasets import load_dataset
from scipy.io.wavfile import write

from tqdm import tqdm

import numpy as np

SEED = 771

set_seed(SEED)



from utils import timing_cuda, generate_offload

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to run. Larger number of samples might give a better estimate of the average time, but it will take longer to run. The real number of samples used will be batch_size*num_samples",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ylacombe/bark-small",
        help="The model path to use (to set bark-small or bark-large).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/home/yoach/bark_optimization/samples/",
        help="The output folder in which to create samples",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="torch.float32",
        help="Precision of torch dtype (torch.float32, torch.float16)"
    )
    parser.add_argument(
        "--voice_preset",
        type=str,
        default=None,
        help="Voice preset to use (defaults to no voice_preset), e.g 'v2/en_speaker_1'."
    )
    
    parser.add_argument(
        "--optimization_type",
        type=str,
        default='no_optimization',
        help="Optimization type to benchmark. For now, must be in ['no_optimization', 'vocos', 'vocos_coarse', 'vocos_coarse_streaming', 'encodec_v2']."
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature. Careful, set the temperature for every sub-models. So if you benchmark the degradation, be careful."
    )
    parser.add_argument(
        "--use_mix_models",
        action="store_true",
        help="Use bark-large weights for everything but the coarse model which will be loaded in small if true.",
    )
    
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=768,
        help="`max_new_tokens` to generate. This argument is equivalent to the `max_new_tokens` argument in `model.generate`.",
    )
    parser.add_argument(
        "--use_bettertransformer",
        action="store_true",
        help="Use bettertransformer.",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    SAMPLE_RATE = 24_000

    if args.num_samples <= 0:
        raise ValueError("num_samples must be superior to 1")
    # dataset
    dataset = load_dataset("kensho/spgispeech", "dev", split="validation", use_auth_token=True)
    dataset = dataset.shuffle(seed=SEED)["transcript"]
    dataset = dataset[:(args.num_samples)]
    
    optimization_type = args.optimization_type
    
    max_new_tokens = args.max_num_tokens
    batch_size = 1
    
    additionnal_kwargs = {}
    
    if optimization_type == "no_optimization":
        model_class = BarkModel
    elif optimization_type == "encodec_v2":
        class NewModel(BarkModel):
            def codec_decode(self, fine_output):
                """Turn quantized audio codes into audio array using encodec."""

                fine_output = fine_output.transpose(0, 1)
                emb = self.mbd.codec_model.model.quantizer.decode(fine_output)
                out = self.mbd.codec_model.model.decoder(emb)
                
                
                wav_diffusion = self.mbd.generate(emb=emb, size=out.size())
                
                wav_diffusion = wav_diffusion.squeeze(1)  # squeeze the codebook dimension

                return wav_diffusion
    
        model_class = NewModel
    elif optimization_type == 'vocos':
        model_class = BarkVocosFine
    elif optimization_type == 'vocos_coarse':
        model_class = BarkVocosCoarse
    elif optimization_type == 'vocos_coarse_streaming':
        model_class = BarkVocosCoarseStream
        
    
    if args.use_mix_models:
        model = model_class.from_pretrained(
                "suno/bark",
                torch_dtype=eval(args.precision),
                low_cpu_mem_usage= (args.precision=="torch.float16"),
                )  
        
        model_small = model_class.from_pretrained(
                "suno/bark-small",
                torch_dtype=eval(args.precision),
                low_cpu_mem_usage= (args.precision=="torch.float16"),
                ).coarse_acoustics
        
        model.coarse_acoustics = model_small

    else:
        model = model_class.from_pretrained(
            args.model_path,
            torch_dtype=eval(args.precision),
            )  
    
    if "vocos" in  optimization_type:
        # hack to make vocos fp32
        model.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(model.device)
    
    if optimization_type == "encodec_v2":
        from audiocraft.models import MultiBandDiffusion
        bandwidth = 3.0  # 1.5, 3.0, 6.0
        del model.codec_model
        model.mbd = MultiBandDiffusion.get_mbd_24khz(bw=bandwidth)
    
    model = model.to("cuda")


    if "vocos" in  optimization_type:
        # hack to make vocos fp32
        model.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(model.device)
        
    if args.use_bettertransformer:
        model = model.to_bettertransformer()

    # processor
    processor = BarkProcessor.from_pretrained(args.model_path)
    

    model_path = args.model_path if not args.use_mix_models else "mixed_models"
    
    flash_string = "flash_attention" if args.use_bettertransformer else "standard_attention"
    precision_string = "fp32" if args.precision == "torch.float32" else "fp16"
    
    folder_name = f"{model_path.replace('-','_').replace('/','_')}_{args.optimization_type}_{flash_string}_{precision_string}"
    folder_path = os.path.join(args.output_folder, folder_name)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    for i in tqdm(range(args.num_samples)):
        set_seed(SEED)
        
        inputs = processor(dataset[i], args.voice_preset).to("cuda")

        output = model.generate(**inputs, temperature=args.temperature,
                                #semantic_max_new_tokens=max_new_tokens,
            )
        
        write(os.path.join(folder_path,f"samples_{i}.wav"), model.generation_config.sample_rate, output.detach().cpu().numpy().squeeze().astype(np.float32))