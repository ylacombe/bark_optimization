import os
import argparse

import torch
import transformers
from transformers import (
    BarkModel,
    BarkProcessor,
)

from bark.api import generate_audio
from bark.generation import preload_models, SAMPLE_RATE

import torch
from transformers import set_seed
from datasets import load_dataset


SEED = 770

set_seed(SEED)



from utils import timing_cuda


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
#    parser.add_argument(
#        "--compile-mode",
#        type=str,
#        default="reduce-overhead",
#        help="The model compilation mode to use. Refer to the official tutorial of torch.compile: https://pytorch.org/tutorials//intermediate/torch_compile_tutorial.html for more details.",
#    )
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
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()


    # dataset
    dataset = load_dataset("kensho/spgispeech", "dev", split="validation", use_auth_token=True)
    dataset = dataset.shuffle(seed=SEED)["transcript"]
    dataset = dataset[:args.num_samples]

    # model
    model = BarkModel.from_pretrained(
        args.model_path,
        torch_dtype=eval(args.precision),
        low_cpu_mem_usage=True,
        )

    # processor
    processor = BarkProcessor.from_pretrained(args.model_path)

    if args.use_cpu:
        model = model.to("cpu")
    else:
        model = model.to("cuda")


    # warmup
    _ = timing_cuda(
        model=model,
        processor=processor,
        num_runs=2,
        input_texts=dataset[:min(2, args.num_samples)],
        voice_preset=args.voice_preset,
        device = model.device,
    )
    

    # real timing
    hf_time, hf_max_memory = timing_cuda(
        model=model,
        processor=processor,
        num_runs=args.num_samples,
        input_texts=dataset,
        voice_preset=args.voice_preset,
        device = model.device,
    ) 

#    model = model.to_bettertransformer()

    # warmup
    _ = timing_cuda(
        model=model,
        processor=processor,
        num_runs=2,
        input_texts=dataset[:min(2, args.num_samples)],
        voice_preset=args.voice_preset,
        device = model.device,
    )

    # real timing
    sdpa_no_compile_time, no_compile_max_memory = timing_cuda(
        model=model,
        processor=processor,
        num_runs=args.num_samples,
        input_texts=dataset,
        voice_preset=args.voice_preset,
        device = model.device,
    ) 

    model = torch.compile(model, mode=args.compile_mode, fullgraph=True, dynamic=True)

    # warmup
    _ = timing_cuda(
        model=model,
        processor=processor,
        num_runs=2,
        input_texts=dataset[:min(2, args.num_samples)],
        voice_preset=args.voice_preset,
        device = model.device,
    )

    # real time
    sdpa_compile_time, compile_max_memory = timing_cuda(
        model=model,
        processor=processor,
        num_runs=args.num_samples,
        input_texts=dataset,
        voice_preset=args.voice_preset,
        device = model.device,
    ) 

    full_header = "pt_version;model_name;compile_mode;batch_size;max_num_tokens;run_type;precision;hf_time;sdpa_no_compile_time;sdpa_compile_time\n"

    if os.path.isfile(args.output_file):
        with open(args.output_file, "r") as f:
            header = f.readline()
        if header != full_header:
            raise ValueError("Output file exists but has incorrect header")
    else:
        with open(args.output_file, "w") as f:
            f.write(full_header)

    with open(args.output_file, "a") as f:
        max_tokens = args.max_num_tokens
        run_type = "forward-only" if not args.run_generate else "generate"
        precision = str(model.dtype)

        f.write(
                f"{torch.__version__};{args.model_name};{args.compile_mode};{args.batch_size};{max_tokens};{run_type};{precision};{round(hf_time, 5)};{round(sdpa_no_compile_time, 5)};{round(sdpa_compile_time, 5)}\n"
        )
