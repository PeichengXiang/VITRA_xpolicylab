import copy
import transformers
import torch

def build_vlm(vlm_config):
    vlm_config = copy.deepcopy(vlm_config)
    model_path = vlm_config.get("pretrained_model_name_or_path")
    model_name = vlm_config.get("name")
    model_type = vlm_config.get("type", "AutoModel")
    if model_name == "paligemma":
        from transformers import (
            PaliGemmaConfig,
            PaliGemmaForConditionalGeneration,
            PaliGemmaProcessor,
        )

        # The released VITRA checkpoint is a strict, full state dict.  Allow the
        # architecture to be created from a local PaliGemma config so training
        # does not download and then immediately overwrite a second 3B weight
        # file.  The default remains the official from_pretrained path.
        if vlm_config.get("initialize_from_config", False):
            config = PaliGemmaConfig.from_pretrained(model_path, local_files_only=True)
            model = PaliGemmaForConditionalGeneration(config)
        else:
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                # attn_implementation="eager",
                # revision="bfloat16",
            )
        processor = PaliGemmaProcessor.from_pretrained(model_path)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return processor, model
