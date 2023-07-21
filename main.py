import os
import argparse

import torch
import transformers
from transformers import (
    BarkModel,
    BarkProcessor,
)

from bark_modified import FlashAttentionBarkModel

from bark.api import generate_audio
from bark.generation import preload_models, SAMPLE_RATE

import torch
from transformers import set_seed
from datasets import load_dataset


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
        help="Optimization type to benchmark. For now, must be in ['flash_attention', 'no_optimization', 'generated_assistant', 'bettertransformer']."
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
    
    if optimization_type in ["no_optimization", "generated_assistant", "bettertransformer"]:
        model_class = BarkModel

    elif optimization_type == "flash_attention":
        model_class = FlashAttentionBarkModel
        
    
        

    if args.precision in ["torch.int4", "torch.int8"]:
        from transformers import BitsAndBytesConfig
        quantization_config = {
            "llm_int8_skip_modules":["encodec"],
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
        model = model_class.from_pretrained(
                args.model_path,
                torch_dtype=eval(args.precision),
                low_cpu_mem_usage= (args.precision=="torch.float16"),
                )  
        
        if args.use_cpu:
            model = model.to("cpu")
        else:
            model = model.to("cuda")

    # processor
    processor = BarkProcessor.from_pretrained(args.model_path)
    
    
    if optimization_type == "generated_assistant":
        if "bark-large" not in args.model_path:
            raise ValueError("When using generated_assistant, you want to use 'bark_large' version.")
          
        model_small = BarkModel.from_pretrained("ylacombe/bark-small",
                                                torch_dtype=eval(args.precision),
            low_cpu_mem_usage= (args.precision=="torch.float16"),)
        
        additionnal_kwargs["coarse_assistant_model"] = model_small.coarse_acoustics.to(model.device)
        
        
    if optimization_type == "bettertransformer":
        
        from optimum.bettertransformer import BetterTransformer


        model = BetterTransformer.transform(model, keep_original_model=False)

    handmade_generate = generate_offload if args.use_offload else None

    # warmup
    _ = timing_cuda(
        model=model,
        processor=processor,
        num_runs=2,
        input_text=dataset[:min(2*batch_size, (args.num_samples*batch_size))],
        voice_preset=args.voice_preset,
        device = torch.device('cuda:0') if args.use_offload else model.device,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        handmade_generate=handmade_generate,
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
        device = torch.device('cuda:0') if args.use_offload else model.device,
        temperature = args.temperature,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        handmade_generate=handmade_generate,
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
        
        f.write(
                f"{torch.__version__},{args.model_path},{offload_string},{batch_size},{max_tokens},{args.optimization_type},{args.num_samples},{args.num_runs},{precision},{round(hf_time, 5)},{hf_max_memory},{round(hf_throughput, 5)},{round(args.temperature,3)}\n"
        )
