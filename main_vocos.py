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

import torch
from datasets import load_dataset

from transformers import set_seed

from vocos import Vocos 

SEED = 771

set_seed(SEED)



from utils import timing_cuda, generate_offload

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_runs",
        type=int,
        default=4,
        help="Number of batches to run. The average time across these runs will be reported. Larger runs might give a better estimate of the average time, but it will take longer to run.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of samples to run. Larger number of samples might give a better estimate of the average time, but it will take longer to run. The real number of samples used will be batch_size*num_samples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Input batch size.",
    )
#    parser.add_argument(
#        "--max-num-tokens",
#        type=int,
#        default=256,
#        help="`max_new_tokens` to generate. This argument is equivalent to the `max_new_tokens` argument in `model.generate`.",
#    ) # TODO: for now, I won't set this because we already have a limit in generation_config (764 in the first stage)
    parser.add_argument(
        "--model_path",
        type=str,
        default="ylacombe/bark-small",
        help="The model path to use (to set bark-small or bark-large).",
    )
    parser.add_argument(
        "--use_offload",
        action="store_true",
        help="Use offload if true.",
    )
    parser.add_argument(
        "--use_mix_models",
        action="store_true",
        help="Use bark-large weights for everything but the coarse model which will be loaded in small if true.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output_v3.csv",
        help="The output file to write results. If the file does not exist, it will be created. If the file exists, the results will be appended to the file.",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use CUDA if available.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="torch.float32",
        help="Precision of torch dtype (torch.float32, torch.float16, torch.int4, torch.int8)"
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
        default=0.7,
        help="Temperature. Careful, set the temperature for every sub-models. So if you benchmark the degradation, be careful."
    )
    
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=768,
        help="`max_new_tokens` to generate. This argument is equivalent to the `max_new_tokens` argument in `model.generate`.",
    )
    
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("num_samples must be superior to 1")
    # dataset
    dataset = load_dataset("kensho/spgispeech", "dev", split="validation", use_auth_token=True)
    dataset = dataset.shuffle(seed=SEED)["transcript"]
    dataset = dataset[:(args.num_samples*args.batch_size)]
    
    optimization_type = args.optimization_type
    
    max_new_tokens = args.max_num_tokens
    batch_size = args.batch_size
    
    additionnal_kwargs = {}
    
    if optimization_type in ["no_optimization"]:
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
    elif optimization_type == "flash_attention":
        model_class = FlashAttentionBarkModel
        
    elif optimization_type == 'vocos':
        model_class = BarkVocosFine
    elif optimization_type == 'vocos_coarse':
        model_class = BarkVocosCoarse
    elif optimization_type == 'vocos_coarse_streaming':
        model_class = BarkVocosCoarseStream
        
    
    if optimization_type == "mixed_precision":

        from transformers import BitsAndBytesConfig
        quantization_config = {
            "llm_int8_skip_modules":["semantic", "coarse_acoustics" ,"codec_model"],
        }

        quantization_config["load_in_4bit"] = True

        quantization_config = BitsAndBytesConfig(**quantization_config)


        model = TestBarkModel.from_pretrained(args.model_path,
                                        torch_dtype=eval("torch.float16"),
                                        quantization_config=quantization_config)


        model.semantic = model.semantic.to("cpu")
        model.coarse_acoustics = model.coarse_acoustics.to("cpu")
        model.codec_model = model.codec_model.to("cpu")


        from optimum.bettertransformer import BetterTransformer


        model = BetterTransformer.transform(model, keep_original_model=False)
        

    elif args.precision in ["torch.int4", "torch.int8"]:
        from transformers import BitsAndBytesConfig
        quantization_config = {
            "llm_int8_skip_modules":[]#["encodec"],
        }
        
        if args.precision == "torch.int4":
            quantization_config["load_in_4bit"] = True
        else:
            quantization_config["load_in_8bit"] = True
        
        
        quantization_config = BitsAndBytesConfig(**quantization_config)
        
        
        model = model_class.from_pretrained(
                args.model_path,
                quantization_config=quantization_config)

        
    else:
        
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
            
            print(model.coarse_acoustics)
            model.coarse_acoustics = model_small
            print(model.coarse_acoustics)
            
        else:
            model = model_class.from_pretrained(
                    args.model_path,
                    torch_dtype=eval(args.precision),
                    low_cpu_mem_usage= (args.precision=="torch.float16"),
                    )  
        
        if optimization_type == "encodec_v2":
            from audiocraft.models import MultiBandDiffusion
            bandwidth = 3.0  # 1.5, 3.0, 6.0
            del model.codec_model
            model.mbd = MultiBandDiffusion.get_mbd_24khz(bw=bandwidth)


        if args.use_cpu:
            model = model.to("cpu")
        else:
            if "vocos" in  optimization_type:
                # hack to make vocos fp32
                model.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(model.device)
            model = model.to("cuda")
            if "vocos" in  optimization_type:
                # hack to make vocos fp32
                model.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(model.device)

        
    # processor
    processor = BarkProcessor.from_pretrained(args.model_path)
    
    
    if optimization_type == "generated_assistant":
        if "bark-large" not in args.model_path:
            raise ValueError("When using generated_assistant, you want to use 'bark_large' version.")
          
        model_small = BarkModel.from_pretrained("ylacombe/bark-small",
                                                torch_dtype=eval(args.precision),
            low_cpu_mem_usage= (args.precision=="torch.float16"),)
        
        additionnal_kwargs["coarse_assistant_model"] = model_small.coarse_acoustics.to(model.device)
        
        
        
    # TODO: attention no_optim = with bettertransform
    from optimum.bettertransformer import BetterTransformer


    model = BetterTransformer.transform(model, keep_original_model=False)
        
    if args.use_offload:
        model.enable_cpu_offload()

    # warmup
    _ = timing_cuda(
        model=model,
        processor=processor,
        num_runs=2,
        input_text=dataset[:min(2*batch_size, (args.num_samples*batch_size))],
        voice_preset=args.voice_preset,
        device = torch.device('cuda') if args.use_offload else model.device,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        **additionnal_kwargs,
    )
        
    #if args.precision not in ["torch.int4", "torch.int8"]:
    #    if args.use_cpu:
    #        model = model.to("cpu")
    #    else:
    #        model = model.to("cuda")

    # real timing
    hf_time, hf_max_memory, hf_throughput = timing_cuda(
        model=model,
        processor=processor,
        num_runs=args.num_runs,
        input_text=dataset,
        voice_preset=args.voice_preset,
        device = torch.device('cuda') if args.use_offload else model.device,
        temperature = args.temperature,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        **additionnal_kwargs,
    ) 


    full_header = "pt_version,model_name,use_offload,batch_size,max_num_tokens,optimization,num_samples,num_runs,precision,latency,max_memory,throughput,temperature\n"

    if os.path.isfile(args.output_file):
        with open(args.output_file, "r") as f:
            header = f.readline()
        if header != full_header:
            raise ValueError("Output file exists but has incorrect header")
    else:
        with open(args.output_file, "w") as f:
            f.write(full_header)

    with open(args.output_file, "a") as f:
        max_tokens = max_new_tokens
        precision = args.precision #str(model.dtype)

        offload_string = "with_offload" if args.use_offload else "no_offload"
        
        model_path = args.model_path if not args.use_mix_models else "mixed_models"
        
        f.write(
                f"{torch.__version__},{model_path},{offload_string},{batch_size},{max_tokens},{args.optimization_type},{args.num_samples},{args.num_runs},{precision},{round(hf_time, 5)},{hf_max_memory},{round(hf_throughput, 5)},{round(args.temperature,3)}\n"
        )
