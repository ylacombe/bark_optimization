import os
import argparse

import torch
import transformers
from transformers import (
    BarkModel,
    BarkProcessor,
)

from bark_modified import AssistedBarkModel, FlashAttentionBarkModel

from bark.api import generate_audio
from bark.generation import preload_models, SAMPLE_RATE

import torch
from transformers import set_seed
from datasets import load_dataset


SEED = 770

set_seed(SEED)



from utils import timing_cuda, timing_cuda_assistant_model


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
        help="Number of samples to run. Larger number of samples might give a better estimate of the average time, but it will take longer to run.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Input batch size. TODO: for now, only batch size = 1",
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
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="The model compilation mode to use. Refer to the official tutorial of torch.compile: https://pytorch.org/tutorials//intermediate/torch_compile_tutorial.html for more details.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output.csv",
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
        help="Precision of torch dtype"
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
        help="Optimization type to benchmark. For now, must be in ['flash_attention', 'no_optimization', 'generated_assistant']."
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature. Careful, set the temperature for every sub-models. So if you benchmark the degradation, be careful."
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
    dataset = dataset[:args.num_samples]
    
    optimization_type = args.optimization_type

    if optimization_type == "no_optimization":
        # model
        model = BarkModel.from_pretrained(
            args.model_path,
            torch_dtype=eval(args.precision),
            low_cpu_mem_usage= (args.precision=="torch.float16"),
            )
    elif optimization_type == "flash_attention":
        model = FlashAttentionBarkModel.from_pretrained(
            args.model_path,
            torch_dtype=eval(args.precision),
            low_cpu_mem_usage= (args.precision=="torch.float16"),
            )
    elif optimization_type == "generated_assistant":
        if "bark-large" not in args.model_path:
            raise ValueError("When using generated_assistant, you want to use 'bark_large' version.")
        
        model = AssistedBarkModel.from_pretrained(
            args.model_path,
            torch_dtype=eval(args.precision),
            low_cpu_mem_usage= (args.precision=="torch.float16"),
            )  
        
        model_small = BarkModel.from_pretrained("ylacombe/bark-small",
                                                torch_dtype=eval(args.precision),
            low_cpu_mem_usage= (args.precision=="torch.float16"),)

    # TODO: compilation
    # processor
    processor = BarkProcessor.from_pretrained(args.model_path)

    if args.use_cpu:
        model = model.to("cpu")
    else:
        model = model.to("cuda")
        
    if optimization_type == "generated_assistant":
        model_small = model_small.to(model.device)
        
        
        # warmup
        _ = timing_cuda_assistant_model(
            model=model,
            processor=processor,
            num_runs=2,
            input_text=dataset[:min(2, args.num_samples)],
            voice_preset=args.voice_preset,
            device = model.device,
            coarse_assistant=model_small.coarse_acoustics,
        )
        

        # real timing
        hf_time, hf_max_memory = timing_cuda_assistant_model(
            model=model,
            processor=processor,
            num_runs=args.num_runs,
            input_text=dataset,
            voice_preset=args.voice_preset,
            device = model.device,
            coarse_assistant=model_small.coarse_acoustics,
            temperature = args.temperature,
        ) 
        
        print("Now compile")
        
        #model = torch.compile(model, mode=args.compile_mode, fullgraph=True, dynamic=True)
        model.semantic = torch.compile(model.semantic, fullgraph=True, dynamic=True, mode = args.compile_mode) 
        model.coarse_acoustics = torch.compile(model.coarse_acoustics, fullgraph=True, dynamic=True, mode = args.compile_mode) 
        model.fine_acoustics = torch.compile(model.fine_acoustics, fullgraph=True, dynamic=True, mode = args.compile_mode)

        #small_coarse = torch.compile(model_small.coarse_acoustics, mode=args.compile_mode, fullgraph=True, dynamic=True)
        small_coarse = model_small.coarse_acoustics
        
        # warmup
        _ = timing_cuda_assistant_model(
            model=model,
            processor=processor,
            num_runs=2,
            input_text=dataset[:min(2, args.num_samples)],
            voice_preset=args.voice_preset,
            device = model.device,
            coarse_assistant=small_coarse,
        )
        

        # real timing
        sdpa_compile_time, compile_max_memory = timing_cuda_assistant_model(
            model=model,
            processor=processor,
            num_runs=args.num_runs,
            input_text=dataset,
            voice_preset=args.voice_preset,
            device = model.device,
            coarse_assistant=small_coarse,
            temperature = args.temperature,
        ) 


    else:


        # warmup
        _ = timing_cuda(
            model=model,
            processor=processor,
            num_runs=2,
            input_text=dataset[:min(2, args.num_samples)],
            voice_preset=args.voice_preset,
            device = model.device,
        )
        

        # real timing
        hf_time, hf_max_memory = timing_cuda(
            model=model,
            processor=processor,
            num_runs=args.num_runs,
            input_text=dataset,
            voice_preset=args.voice_preset,
            device = model.device,
            temperature = args.temperature,
        ) 

        print("now compile")
        #model = torch.compile(model, mode=args.compile_mode, fullgraph=True, dynamic=True)
        model.semantic = torch.compile(model.semantic, fullgraph=True, dynamic=True, mode = args.compile_mode)
        model.coarse_acoustics = torch.compile(model.coarse_acoustics, fullgraph=True, dynamic=True, mode = args.compile_mode)
        model.fine_acoustics = torch.compile(model.fine_acoustics, fullgraph=True, dynamic=True, mode = args.compile_mode)

        # warmup
        _ = timing_cuda(
            model=model,
            processor=processor,
            num_runs=2,
            input_text=dataset[:min(2, args.num_samples)],
            voice_preset=args.voice_preset,
            device = model.device,
        )

        # real time
        sdpa_compile_time, compile_max_memory = timing_cuda(
            model=model,
            processor=processor,
            num_runs=args.num_runs,
            input_text=dataset,
            voice_preset=args.voice_preset,
            device = model.device,
            temperature = args.temperature,
        ) 

    full_header = "pt_version,model_name,compile_mode,batch_size,max_num_tokens,optimization,num_samples,num_runs,precision,hf_time,hf_max_memory,sdpa_compile_time,compile_max_memory,temperature\n"

    if os.path.isfile(args.output_file):
        with open(args.output_file, "r") as f:
            header = f.readline()
        if header != full_header:
            raise ValueError("Output file exists but has incorrect header")
    else:
        with open(args.output_file, "w") as f:
            f.write(full_header)

    with open(args.output_file, "a") as f:
        max_tokens = 764#args.max_num_tokens
        precision = str(model.dtype)

        f.write(
                f"{torch.__version__},{args.model_path},{args.compile_mode},{args.batch_size},{max_tokens},{args.optimization_type},{args.num_samples},{args.num_runs},{precision},{round(hf_time, 5)},{hf_max_memory},{round(sdpa_compile_time, 5)},{compile_max_memory},{round(args.temperature,3)}\n"
        )
