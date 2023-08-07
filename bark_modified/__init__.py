#from .bark_assisted_generation import BarkModel as AssistedBarkModel
from .bark_flash_attention import BarkModel as FlashAttentionBarkModel

#from .bark_flash_attention_2 import BarkModel as FlashAttention2BarkModel


from .bark_offload import BarkModel as OffloadBarkModel


from .bark_best_combination import BarkModel as TestBarkModel


from .bark_vocos import BarkModel as BarkVocosFine
from .bark_vocos_coarse import BarkModel as BarkVocosCoarse
from .bark_vocos_coarse_stream import BarkModel as BarkVocosCoarseStream
