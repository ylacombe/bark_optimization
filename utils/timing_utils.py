import torch

from typing import Tuple, List
from tqdm import tqdm

from transformers.models.bark.generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkSemanticGenerationConfig,
)

def timing_cuda_step_by_step(
    model: torch.nn.Module,
    processor: "BarkProcessor",
    num_runs: int,
    input_texts: List[str],
    voice_preset: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, int]:
    """test generate from BarkModel steps (generate_text_semantic, generate_coarse, generate_fine)
    """
    
    inputs = processor(input_texts, voice_preset).to(model.device)
    
    input_ids = inputs["input_ids"]
    history_prompt = inputs.get("history_prompt", None)
    
    semantic_generation_config = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
    coarse_generation_config = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
    fine_generation_config = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)

    # SEMANTIC
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()
    
    for i in tqdm(range(len(input_texts))):
        for _ in range(num_runs):
            semantic_output = model.semantic.generate_text_semantic(
                input_ids[[i]],
                history_prompt=history_prompt,
                attention_mask=None,
                semantic_generation_config=semantic_generation_config,
            )


    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)
    
    semantic_time, semantic_memory =  (start_event.elapsed_time(end_event) * 1.0e-3) / (num_runs * len(input_texts)), max_memory
    
    # create input for next step
    semantic_output = []
    for i in tqdm(range(len(input_texts))):
        semantic_output.append(model.semantic.generate_text_semantic(
            input_ids[[i]],
            history_prompt=history_prompt,
            attention_mask=None,
            semantic_generation_config=semantic_generation_config,
        ))



    # COARSE
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()
    
    for i in tqdm(range(len(input_texts))):
        for _ in range(num_runs):
            coarse_output = model.coarse_acoustics.generate_coarse(
                semantic_output[i],
                history_prompt=history_prompt,
                semantic_generation_config=semantic_generation_config,
                coarse_generation_config=coarse_generation_config,
                codebook_size=model.generation_config.codebook_size,
            )


    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)
    
    coarse_time, coarse_memory =  (start_event.elapsed_time(end_event) * 1.0e-3) / (num_runs * len(input_texts)), max_memory
    
    
    
    # create input for next step
    coarse_output = []
    for i in tqdm(range(len(input_texts))):
        coarse_output.append(model.coarse_acoustics.generate_coarse(
                semantic_output[i],
                history_prompt=history_prompt,
                semantic_generation_config=semantic_generation_config,
                coarse_generation_config=coarse_generation_config,
                codebook_size=model.generation_config.codebook_size,
            ))

    
    
    # FINE
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()
    
    for i in tqdm(range(len(input_texts))):
        for _ in range(num_runs):
            output = model.fine_acoustics.generate_fine(
                coarse_output[i],
                history_prompt=history_prompt,
                semantic_generation_config=semantic_generation_config,
                coarse_generation_config=coarse_generation_config,
                fine_generation_config=fine_generation_config,
                codebook_size=model.generation_config.codebook_size,
            )


    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)
    
    fine_time, fine_memory =  (start_event.elapsed_time(end_event) * 1.0e-3) / (num_runs * len(input_texts)), max_memory
    
    
    # create input for next step
    output = []
    for i in range(len(input_texts)):
        output.append(model.fine_acoustics.generate_fine(
                coarse_output[i],
                history_prompt=history_prompt,
                semantic_generation_config=semantic_generation_config,
                coarse_generation_config=coarse_generation_config,
                fine_generation_config=fine_generation_config,
                codebook_size=model.generation_config.codebook_size,
            ))

    
    # CODEC_DECODE
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()
    
    for i in range(len(input_texts)):
        for _ in tqdm(range(num_runs)):
            _ = model.codec_decode(output[i])


    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)
    
    codec_time, codec_memory =  (start_event.elapsed_time(end_event) * 1.0e-3) / (num_runs * len(input_texts)), max_memory
    
    
    

    return semantic_time, semantic_memory, coarse_time, coarse_memory, fine_time, fine_memory, codec_time, codec_memory


def timing_cuda(
    model: torch.nn.Module,
    processor: "BarkProcessor",
    num_runs: int,
    input_text: torch.LongTensor,
    voice_preset: str,
    device: torch.device = torch.device("cpu"),
    temperature = 0.7,
    max_new_tokens = 768,
    batch_size = 1,
    handmade_generate = None,
    **kwargs,
) -> Tuple[float, int]:
    """test generate from BarkModel all at once, processing including
    """
    model.generation_config.semantic_config["max_new_tokens"]=max_new_tokens
    
    num_iterations = len(input_text) // batch_size
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()
    
    inputs = processor(input_text, voice_preset).to(device)
    input_ids = inputs["input_ids"]
    history_prompt = inputs.get("history_prompt", None)
    
    
    if handmade_generate is None:
        for i in tqdm(range(num_iterations)):
            for _ in range(num_runs):
                _ = model.generate(input_ids = input_ids[i*batch_size: (i+1)*batch_size], history_prompt=history_prompt, temperature=temperature,
                                **kwargs,
                                        )
    else:
        for i in tqdm(range(num_iterations)):
            for _ in range(num_runs):
                _ = handmade_generate(model, input_ids = input_ids[i*batch_size: (i+1)*batch_size], history_prompt=history_prompt, temperature=temperature,
                                **kwargs,
                                        )
        

    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)
    
    elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3

    latency = elapsed_time / (num_runs * num_iterations)
    throughput = (num_runs * num_iterations * batch_size) / elapsed_time

    return latency, max_memory, throughput


def timing_cuda_old(
    model: torch.nn.Module,
    num_runs: int,
    inputs: torch.LongTensor,
    generation_config: "GenerationConfig" = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, int]:
    """ version from the repo I took (https://github.com/younesbelkada/hf-torch-compile-benchmark/blob/main/main.py)
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start_event.record()

    for _ in tqdm(range(num_runs)):
        if generation_config is not None:
            _ = model.generate(inputs, generation_config=generation_config)
        else:
            kwargs = {"attention_mask": torch.ones_like(inputs, dtype=torch.bool)}
            if model.config.is_encoder_decoder:
                shape = inputs.shape
                if model.config.model_type == "whisper":
                    shape = (inputs.shape[0], model.config.max_target_positions)
                
                kwargs["decoder_input_ids"] = torch.ones(shape, dtype=torch.long, device=inputs.device)
            _ = model(inputs, **kwargs)

    end_event.record()
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)

    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_runs, max_memory