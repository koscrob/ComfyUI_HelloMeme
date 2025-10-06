# coding: utf-8

"""
@File   : hm_adapters.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 5/19/2025
@Desc   : 
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint

from typing import Optional, Union, Dict, List, Tuple
import copy
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from diffusers.configuration_utils import FrozenDict
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
)

from diffusers.utils import logging
from diffusers.utils.constants import USE_PEFT_BACKEND
from diffusers.utils.peft_utils import scale_lora_layers, unscale_lora_layers
from diffusers.utils.import_utils import (
    is_torch_version,
    is_peft_available,
    is_peft_version,
    is_transformers_available,
    is_transformers_version,
)

from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder

from .hm_blocks import (
    SKReferenceAttentionV3,
    SKReferenceAttentionV5,
    STKReferenceModule,
    SKReferenceAttention,
    SKMotionModule,
    SKMotionModuleV5,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True

class HMReferenceAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
                 down_block_types: Tuple[str, ...] = (
                         "CrossAttnDownBlock2D",
                         "CrossAttnDownBlock2D",
                         "CrossAttnDownBlock2D",
                         "DownBlock2D",
                 ),
                 up_block_types: Tuple[str, ...] = (
                         "UpBlock2D",
                         "CrossAttnUpBlock2D",
                         "CrossAttnUpBlock2D",
                         "CrossAttnUpBlock2D"),
                 num_attention_heads: Union[int, Tuple[int, ...]] = 8,
                 version='v1'
                 ):
        super().__init__()

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if version == 'v1':
            self.reference_modules_down = nn.ModuleList([])
            for i, down_block_type in enumerate(down_block_types):
                output_channel = block_out_channels[i]

                self.reference_modules_down.append(
                    SKReferenceAttention(
                        in_channels=output_channel,
                        num_attention_heads=num_attention_heads[i],
                    )
                )

            self.reference_modules_mid = SKReferenceAttention(
                in_channels=block_out_channels[-1],
                num_attention_heads=num_attention_heads[-1],
            )

        self.reference_modules_up = nn.ModuleList([])

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            if i > 0:
                self.reference_modules_up.append(
                    SKReferenceAttention(
                        in_channels=prev_output_channel,
                        num_attention_heads=reversed_num_attention_heads[i],
                        num_positional_embeddings=64 * 2
                    )
                )


class HM3ReferenceAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, block_down_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
                     block_up_channels: Tuple[int, ...] = (1280, 1280, 1280, 640),
                     num_attention_heads: int = 8,
                     use_3d: bool = False):
        super().__init__()

        self.reference_modules_up = nn.ModuleList([])
        for i, in_channels in enumerate(block_up_channels):
            self.reference_modules_up.append(
                SKReferenceAttentionV3(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                    num_positional_embeddings=64*2
                )
            )


class HM5bReferenceAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, block_down_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
                     block_up_channels: Tuple[int, ...] = (1280, 1280, 1280, 640),
                     num_attention_heads: int = 8,
                     use_3d: bool = False):
        super().__init__()

        self.reference_modules_up = nn.ModuleList([])
        for i, in_channels in enumerate(block_up_channels):
            self.reference_modules_up.append(
                STKReferenceModule(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                )
            )


class HM5ReferenceAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, block_down_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
                     block_up_channels: Tuple[int, ...] = (1280, 1280, 1280, 640),
                     num_attention_heads: int = 8,
                     use_3d: bool = False):
        super().__init__()

        self.reference_modules_up = nn.ModuleList([])
        for i, in_channels in enumerate(block_up_channels):
            self.reference_modules_up.append(
                SKReferenceAttentionV5(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                )
            )


class HM3MotionAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,  block_down_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
                     block_up_channels: Tuple[int, ...] = (1280, 1280, 1280, 640),
                     num_attention_heads: int = 8,
                     use_3d: bool = True):
        super().__init__()
        blocks_time_embed_dim = 1280
        self.motion_down = nn.ModuleList([])

        for i, in_channels in enumerate(block_down_channels):
            self.motion_down.append(
                SKMotionModule(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                    blocks_time_embed_dim=blocks_time_embed_dim,
                )
            )

        self.motion_up = nn.ModuleList([])
        for i, in_channels in enumerate(block_up_channels):
            self.motion_up.append(
                SKMotionModule(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                    blocks_time_embed_dim=blocks_time_embed_dim,
                )
            )


class HM5MotionAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,  block_down_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
                     block_up_channels: Tuple[int, ...] = (1280, 1280, 1280, 640),
                     num_attention_heads: int = 8):
        super().__init__()
        self.motion_down = nn.ModuleList([])

        for i, in_channels in enumerate(block_down_channels):
            self.motion_down.append(
                SKMotionModuleV5(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                )
            )

        self.motion_up = nn.ModuleList([])
        for i, in_channels in enumerate(block_up_channels):
            self.motion_up.append(
                SKMotionModuleV5(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                )
            )


class HMPipeline(StableDiffusionImg2ImgPipeline):
    @torch.no_grad()
    def load_lora_weights_sk(self, unet, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
                          adapter_name=None, text_encoder=None, lora_scale=1.0, **kwargs):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=unet,
            adapter_name=adapter_name,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        if not text_encoder is None:
            self.load_lora_into_text_encoder(
                state_dict,
                network_alphas=network_alphas,
                text_encoder=text_encoder,
                lora_scale=lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

    def encode_prompt_sk(
        self,
        text_encoder,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(text_encoder , lora_scale)
            else:
                scale_lora_layers(text_encoder , lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder .text_model.final_layer_norm(prompt_embeds)

        if text_encoder  is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(text_encoder .config, "use_attention_mask") and text_encoder .config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = text_encoder (
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if text_encoder  is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(text_encoder , lora_scale)

        return prompt_embeds, negative_prompt_embeds


# https://github.com/huggingface/diffusers/blob/82058a5413ca09561cc5cc236c4abc5eeda7b209/src/diffusers/loaders/ip_adapter.py

if is_transformers_available():
    from diffusers.models.attention_processor import (
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0,
    )

class CopyWeights(object):
    @classmethod
    def from_unet2d(cls, unet: UNet2DConditionModel):
        # adapted from :https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_motion_model.py

        config = dict(unet.config)

        # Need this for backwards compatibility with UNet2DConditionModel checkpoints
        # if not config.get("num_attention_heads"):
        #     config["num_attention_heads"] = config["attention_head_dim"]

        config = FrozenDict(config)
        model = cls.from_config(config)

        model.conv_in.load_state_dict(unet.conv_in.state_dict())

        model.time_proj.load_state_dict(unet.time_proj.state_dict())
        model.time_embedding.load_state_dict(unet.time_embedding.state_dict())

        if any(
            isinstance(proc, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0))
            for proc in unet.attn_processors.values()
        ):
            attn_procs = {}
            for name, processor in unet.attn_processors.items():
                if name.endswith("attn1.processor"):
                    attn_processor_class = (
                        AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
                    )
                    attn_procs[name] = attn_processor_class()
                else:
                    attn_processor_class = (
                        IPAdapterAttnProcessor2_0
                        if hasattr(F, "scaled_dot_product_attention")
                        else IPAdapterAttnProcessor
                    )
                    attn_procs[name] = attn_processor_class(
                        hidden_size=processor.hidden_size,
                        cross_attention_dim=processor.cross_attention_dim,
                        scale=processor.scale,
                        num_tokens=processor.num_tokens,
                    )
            for name, processor in model.attn_processors.items():
                if name not in attn_procs:
                    attn_procs[name] = processor.__class__()
            model.set_attn_processor(attn_procs)
            model.config.encoder_hid_dim_type = "ip_image_proj"
            model.encoder_hid_proj = unet.encoder_hid_proj

        for i, down_block in enumerate(unet.down_blocks):
            model.down_blocks[i].resnets.load_state_dict(down_block.resnets.state_dict())
            if hasattr(model.down_blocks[i], "attentions"):
                model.down_blocks[i].attentions.load_state_dict(down_block.attentions.state_dict())
            if model.down_blocks[i].downsamplers:
                model.down_blocks[i].downsamplers.load_state_dict(down_block.downsamplers.state_dict())

        for i, up_block in enumerate(unet.up_blocks):
            model.up_blocks[i].resnets.load_state_dict(up_block.resnets.state_dict())
            if hasattr(model.up_blocks[i], "attentions"):
                model.up_blocks[i].attentions.load_state_dict(up_block.attentions.state_dict())
            if model.up_blocks[i].upsamplers:
                model.up_blocks[i].upsamplers.load_state_dict(up_block.upsamplers.state_dict())

        model.mid_block.resnets.load_state_dict(unet.mid_block.resnets.state_dict())
        model.mid_block.attentions.load_state_dict(unet.mid_block.attentions.state_dict())

        if unet.conv_norm_out is not None:
            model.conv_norm_out.load_state_dict(unet.conv_norm_out.state_dict())
        if unet.conv_act is not None:
            model.conv_act.load_state_dict(unet.conv_act.state_dict())
        model.conv_out.load_state_dict(unet.conv_out.state_dict())

        # ensure that the Motion UNet is the same dtype as the UNet2DConditionModel
        model.to(unet.dtype)

        return model

class InsertReferenceAdapter(object):
    def __init__(self):
        self.reference_modules_down = None
        self.reference_modules_mid = None
        self.reference_modules_up = None
        self.motion_down = None
        self.motion_up = None

    def insert_reference_adapter(self, adapter):
        if hasattr(adapter, "reference_modules_down"):
            self.reference_modules_down = copy.deepcopy(adapter.reference_modules_down)
        if hasattr(adapter, "reference_modules_mid"):
            self.reference_modules_mid = copy.deepcopy(adapter.reference_modules_mid)
        if hasattr(adapter, "reference_modules_up"):
            self.reference_modules_up = copy.deepcopy(adapter.reference_modules_up)
        if hasattr(adapter, "motion_down"):
            self.motion_down = copy.deepcopy(adapter.motion_down)
        if hasattr(adapter, "motion_up"):
            self.motion_up = copy.deepcopy(adapter.motion_up)