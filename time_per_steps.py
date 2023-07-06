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

from utils import timing_cuda_step_by_step

from datasets import load_dataset


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
#    )
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
        "--output_file",
        type=str,
        default="per_steps_results.csv",
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
        help="Precision of torch dtype (not supported yet)"
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

    SEED = 770
    NUM_SAMPLES = args.num_samples
    MODEL_PATH = args.model_path
    NUM_RUNS = args.num_runs
    VOICE_PRESET = args.voice_preset
    USE_CPU = args.use_cpu
    OUTPUT_FILE = args.output_file

    COMPILE_MODE = "no mode" 
    BATCH_SIZE = args.batch_size




    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    # TODO: batch

    set_seed(SEED)


    # dataset
    dataset = load_dataset("kensho/spgispeech", "dev", split="validation", use_auth_token=True)
    dataset = dataset.shuffle(seed=SEED)["transcript"]
    dataset = dataset[:NUM_SAMPLES]

    # model
    model = BarkModel.from_pretrained(MODEL_PATH)

    # processor
    processor = BarkProcessor.from_pretrained(MODEL_PATH)

    if USE_CPU:
        model = model.to("cpu")
    else:
        model = model.to("cuda")



    # TODO: do warmup and real timing

    # warmup
    _, _, _, _, _, _, _, _ = timing_cuda_step_by_step(
        model=model,
        processor=processor,
        num_runs=2,
        input_texts=dataset[:min(2, NUM_SAMPLES)],
        voice_preset=VOICE_PRESET,
        device = model.device,
    ) 

    # real timing
    semantic_time, semantic_memory, coarse_time, coarse_memory, fine_time, fine_memory, codec_time, codec_memory = timing_cuda_step_by_step(
        model=model,
        processor=processor,
        num_runs=NUM_RUNS,
        input_texts=dataset,
        voice_preset=VOICE_PRESET,
        device = model.device,
    ) 





    full_header = "pt_version,model_name,compile_mode,batch_size,num_samples,nb_iteration_per_samples,precision,semantic_time,semantic_memory,coarse_time,coarse_memory,fine_time,fine_memory,codec_time,codec_memory\n"

    if os.path.isfile(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            header = f.readline()
        if header != full_header:
            raise ValueError("Output file exists but has incorrect header")
    else:
        with open(OUTPUT_FILE, "w") as f:
            f.write(full_header)

    with open(OUTPUT_FILE, "a") as f:
        precision = str(model.dtype)

        f.write(
                f"{torch.__version__},{MODEL_PATH},{COMPILE_MODE},{BATCH_SIZE},{NUM_SAMPLES},{NUM_RUNS},{precision},{round(semantic_time, 5)},{round(semantic_memory, 5)},{round(coarse_time, 5)},{round(coarse_memory, 5)},{round(fine_time, 5)},{round(fine_memory, 5)},{round(codec_time, 5)},{round(codec_memory, 5)}\n"
        )
