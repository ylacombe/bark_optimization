
from queue import Queue
from typing import TYPE_CHECKING, Optional

from vocos import Vocos

from transformers.generation.streamers import BaseStreamer
import torch
from threading import Thread
from transformers.models.bark.generation_configuration_bark import BarkGenerationConfig, BarkSemanticGenerationConfig, BarkCoarseGenerationConfig, BarkFineGenerationConfig

import numpy as np

# ATTENTION: Not full code, it's in test_streaming.py
class BarkAudioStreamer(BaseStreamer):
    """
    Streamer that stores print-ready text in a queue, to be used by a downstream application as an iterator. This is
    useful for applications that benefit from acessing the generated text in a non-blocking way (e.g. in an interactive
    Gradio demo).

    <Tip warning={true}>

    The API for the streamer classes is still under development and may change in the future.

    </Tip>

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenized used to decode the tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt to `.generate()` or not. Useful e.g. for chatbots.
        timeout (`float`, *optional*):
            The timeout for the text queue. If `None`, the queue will block indefinitely. Useful to handle exceptions
            in `.generate()`, when it is called in a separate thread.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        >>> from threading import Thread

        >>> tok = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> inputs = tok(["An increasing sequence: one,"], return_tensors="pt")
        >>> streamer = TextIteratorStreamer(tok)

        >>> # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        >>> generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        >>> thread = Thread(target=model.generate, kwargs=generation_kwargs)
        >>> thread.start()
        >>> generated_text = ""
        >>> for new_text in streamer:
        ...     generated_text += new_text
        >>> generated_text
        'An increasing sequence: one, two, three, four, five, six, seven, eight, nine, ten, eleven,'
        ```
    """

    def __init__(self, timeout: Optional[float] = None, device = None ):


        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout
        
        self.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)
        self.bandwidth_id = torch.tensor([2]).to(device)

    def put(self, value):
        """
        Receives coarse values, decodes them, and prints them to stdout as soon as they form entire words.
        """
        # TODO: for now no speaker embeddings
        
        # TODO: not by hand
        n_coarse_codebooks = 2 #coarse_generation_config.n_coarse_codebooks
        semantic_vocab_size = 10000 #semantic_generation_config.semantic_vocab_size
        codebook_size = 1024 #codebook_size

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("BarkAudioStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
            
        # reshape (seq_len, n_coarse_codebooks)
        value = value.view(-1, n_coarse_codebooks)

        # brings ids into the range [0, codebook_size -1]
        value = torch.remainder(value - semantic_vocab_size, codebook_size)
        
        # transpose
        value = value.transpose(0,1)
        
        value = self.vocos.codes_to_features(value)

        value = self.vocos.decode(value, bandwidth_id=self.bandwidth_id).squeeze().cpu().numpy()
        
        self.on_finalized_audio(value)

    def end(self):
        """End stream."""
        # HOW TO END ?
        pass
    

    def on_finalized_audio(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.audio_queue.put(text, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        
        
        if not isinstance(value, np.ndarray):
            raise StopIteration()
        else:
            return value


from transformers import AutoModel, AutoProcessor
from transformers import set_seed
set_seed(0)

def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return device

device = _grab_best_device()

processor = AutoProcessor.from_pretrained("ylacombe/bark-small")
sentences = [
    "I am not sure how to feel about it, I've got a bad feeling about it. The force is strong here.",
]
tokenized_text = processor(sentences, None).to(device)
SAMPLE_RATE = 24_000


HUB_PATH = "suno/bark-small"

# import model
bark = AutoModel.from_pretrained(HUB_PATH, torch_dtype=torch.float16).to(device)

# convert to bettertransformer
bark = bark.to_bettertransformer()


streamer = BarkAudioStreamer(timeout=20, device=device)

SLIDING_WINDOW_LEN = 60

# Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
generation_kwargs = dict(self = bark, **tokenized_text, streamer=streamer, semantic_max_new_tokens=400)

def generate_coarse(
    self,
    input_ids = None,
    history_prompt = None,
    streamer = None,
    **kwargs,
) -> torch.LongTensor:

    semantic_generation_config = BarkSemanticGenerationConfig(**self.generation_config.semantic_config)
    coarse_generation_config = BarkCoarseGenerationConfig(**self.generation_config.coarse_acoustics_config)
    fine_generation_config = BarkFineGenerationConfig(**self.generation_config.fine_acoustics_config)
    
    coarse_generation_config.sliding_window_len = SLIDING_WINDOW_LEN
    
    kwargs_semantic = {
        # if "attention_mask" is set, it should not be passed to CoarseModel and FineModel
        "attention_mask": kwargs.pop("attention_mask", None)
    }
    kwargs_coarse = {}
    kwargs_fine = {}
    for key, value in kwargs.items():
        if key.startswith("semantic_"):
            key = key[len("semantic_") :]
            kwargs_semantic[key] = value
        elif key.startswith("coarse_"):
            key = key[len("coarse_") :]
            kwargs_coarse[key] = value
        elif key.startswith("fine_"):
            key = key[len("fine_") :]
            kwargs_fine[key] = value
        else:
            # If the key is already in a specific config, then it's been set with a
            # submodules specific value and we don't override
            if key not in kwargs_semantic:
                kwargs_semantic[key] = value
            if key not in kwargs_coarse:
                kwargs_coarse[key] = value
            if key not in kwargs_fine:
                kwargs_fine[key] = value

    # 1. Generate from the semantic model
    semantic_output = self.semantic.generate(
        input_ids,
        history_prompt=history_prompt,
        semantic_generation_config=semantic_generation_config,
        **kwargs_semantic,
    )

    # 2. Generate from the coarse model
    coarse_output = self.coarse_acoustics.generate(
        semantic_output,
        history_prompt=history_prompt,
        semantic_generation_config=semantic_generation_config,
        coarse_generation_config=coarse_generation_config,
        codebook_size=self.generation_config.codebook_size,
        streamer = streamer,
        **kwargs_coarse,
    )

    return coarse_output

thread = Thread(target=generate_coarse, kwargs=generation_kwargs)
thread.start()


import socket

# Audio settings
SAMPLE_RATE = 24000
CHUNK_SIZE = 8192

# Create a socket server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(('0.0.0.0', 5555))  # Bind to all available network interfaces
server_socket.listen(1)  # Listen for one incoming connection

print("Listening for incoming connections...")

connection, client_address = server_socket.accept()
print("Connected to:", client_address)

for new_audio in streamer:
    # Send audio data to the client
    print(new_audio.shape)
    connection.sendall(new_audio.tostring())