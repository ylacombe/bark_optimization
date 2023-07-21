from transformers.models.bark.generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkSemanticGenerationConfig,
)



def generate_offload(self, 
                     input_ids,
                     history_prompt = None,
                     **kwargs):
    
        semantic_generation_config = BarkSemanticGenerationConfig(**self.generation_config.semantic_config)
        coarse_generation_config = BarkCoarseGenerationConfig(**self.generation_config.coarse_acoustics_config)
        fine_generation_config = BarkFineGenerationConfig(**self.generation_config.fine_acoustics_config)

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

        self.semantic = self.semantic.to("cuda")
        
        # 1. Generate from the semantic model
        semantic_output = self.semantic.generate(
            input_ids,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            **kwargs_semantic,
        )
        
        self.semantic = self.semantic.to("cpu")
        
        
        self.coarse_acoustics = self.coarse_acoustics.to("cuda")

        # 2. Generate from the coarse model
        coarse_output = self.coarse_acoustics.generate(
            semantic_output,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            codebook_size=self.generation_config.codebook_size,
            **kwargs_coarse,
        )
        
        self.coarse_output = self.coarse_acoustics.to("cpu")


        self.fine_acoustics = self.fine_acoustics.to("cuda")

        # 3. "generate" from the fine model
        output = self.fine_acoustics.generate(
            coarse_output,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            fine_generation_config=fine_generation_config,
            codebook_size=self.generation_config.codebook_size,
            **kwargs_fine,
        )
        
        self.fine_acoustics = self.fine_acoustics.to("cpu")

        self.codec_model = self.codec_model.to("cuda")

        # 4. Decode the output and generate audio array
        audio = self.codec_decode(output)
        
        
        self.codec_model = self.codec_model.to("cpu")

        return audio

    