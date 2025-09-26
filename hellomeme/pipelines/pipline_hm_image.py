# coding: utf-8

"""
@File   : hm_pipline_image.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/29/2024
@Desc   :
adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import comfy.utils

from diffusers import EulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps, retrieve_latents

from ..models import HMDenoising3D, HMControlNet, HMControlNet2, HMV2ControlNet, HMV2ControlNet2, HMPipeline
from ..models import HMReferenceAdapter
from ..tools.utils import creat_model_from_cloud


class HMImagePipeline(HMPipeline):
    def caryomitosis(self, **kwargs):
        if hasattr(self, "unet_ref"):
            del self.unet_ref
        self.unet_ref = HMDenoising3D.from_unet2d(self.unet)
        self.unet_ref.cpu()

        if not isinstance(self.unet, HMDenoising3D):
            unet = HMDenoising3D.from_unet2d(unet=self.unet)
            # todo: 不够优雅
            del self.unet
            self.unet = unet
            self.unet.cpu()

        self.vae.cpu()
        self.vae_decode = copy.deepcopy(self.vae)
        self.text_encoder.cpu()
        self.text_encoder_ref = copy.deepcopy(self.text_encoder)
        if hasattr(self, 'safety_checker'):
            del self.safety_checker

    def insert_hm_modules(self, version, dtype, modelscope=False):
        self.version = version

        if version == 'v1':
            hm_reference_dir = 'songkey/hm_reference'
            hm_control_dir = 'songkey/hm_control'
            hm_control2_dir = 'songkey/hm_control2'
        else:
            hm_reference_dir = 'songkey/hm2_reference'
            hm_control_dir = 'songkey/hm2_control'
            hm_control2_dir = 'songkey/hm2_control2'

        if isinstance(self.unet, HMDenoising3D):
            hm_adapter = creat_model_from_cloud(HMReferenceAdapter, hm_reference_dir, modelscope=modelscope)
            self.unet.insert_reference_adapter(hm_adapter)
            self.unet.to(device='cpu', dtype=dtype).eval()

        if hasattr(self, "unet_ref"):
            self.unet_ref.to(device='cpu', dtype=dtype).eval()

        if hasattr(self, "mp_control"):
            del self.mp_control
        if version == 'v1':
            self.mp_control = creat_model_from_cloud(HMControlNet, hm_control_dir, modelscope=modelscope)
        else:
            self.mp_control = creat_model_from_cloud(HMV2ControlNet, hm_control_dir, modelscope=modelscope)
        self.mp_control.to(device='cpu', dtype=dtype).eval()

        if hasattr(self, "mp_control2"):
            del self.mp_control2
        if version == 'v1':
            self.mp_control2 = creat_model_from_cloud(HMControlNet2, hm_control2_dir, modelscope=modelscope)
        else:
            self.mp_control2 = creat_model_from_cloud(HMV2ControlNet2, hm_control2_dir, modelscope=modelscope)
        self.mp_control2.to(device='cpu', dtype=dtype).eval()

        self.vae.to(device='cpu', dtype=dtype).eval()
        self.vae_decode.to(device='cpu', dtype=dtype).eval()
        self.text_encoder.to(device='cpu', dtype=dtype).eval()
        self.text_encoder_ref.to(device='cpu', dtype=dtype).eval()

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            image: PipelineImageInput = None,
            drive_params: Dict[str, Any] = None,
            strength: float = 0.8,
            num_inference_steps: Optional[int] = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: Optional[float] = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = "pil",
            device: Optional[str] = "cpu",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: int = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        num_images_per_prompt = 1

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self.device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        self.text_encoder_ref.to(device=device)
        prompt_embeds_ref, negative_prompt_embeds_ref = self.encode_prompt_sk(
            self.text_encoder_ref,
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        self.text_encoder_ref.cpu()

        self.text_encoder.to(device=device)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt_sk(
            self.text_encoder,
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        self.text_encoder.cpu()

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds_ref = torch.cat([negative_prompt_embeds_ref, prompt_embeds_ref])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Preprocess
        image = self.image_processor.preprocess(image).to(device=device, dtype=prompt_embeds.dtype)

        scheduler = EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )

        # 5. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps, sigmas
        )
        # timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare reference latents
        self.vae.to(device=device)
        ref_latents = [
            retrieve_latents(self.vae.encode(image[i: i + 1].to(device=device)), generator=generator)
            for i in range(batch_size)
        ]
        self.vae.cpu()

        ref_latents = torch.cat(ref_latents, dim=0)
        ref_latents = self.vae.config.scaling_factor * ref_latents
        c, h, w = ref_latents.shape[1:]

        condition = drive_params['condition'].clone().to(device=device)
        if self.do_classifier_free_guidance:
            condition = torch.cat([torch.ones_like(condition) * -1, condition], dim=0)

        control_latents = {}
        if 'drive_coeff' in drive_params:
            self.mp_control.to(device=device)
            drive_coeff = drive_params['drive_coeff'].clone().to(device=device)
            face_parts = drive_params['face_parts'].clone().to(device=device)
            if self.do_classifier_free_guidance:
                drive_coeff = torch.cat([torch.zeros_like(drive_coeff), drive_coeff], dim=0)
                face_parts = torch.cat([torch.zeros_like(face_parts), face_parts], dim=0)
            control_latents1 = self.mp_control(condition=condition, drive_coeff=drive_coeff, face_parts=face_parts)
            control_latents.update(control_latents1)
            self.mp_control.cpu()

        if 'pd_fpg' in drive_params:
            self.mp_control2.to(device=device)
            pd_fpg = drive_params['pd_fpg'].clone().to(device=device)
            if self.do_classifier_free_guidance:
                neg_pd_fpg = drive_params['neg_pd_fpg'].clone().to(device=device)
                neg_pd_fpg.repeat_interleave(pd_fpg.size(1), dim=1)
                pd_fpg = torch.cat([neg_pd_fpg, pd_fpg], dim=0)
            control_latents2 = self.mp_control2(condition=condition, emo_embedding=pd_fpg)
            control_latents.update(control_latents2)
            self.mp_control2.cpu()

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        latent_model_input = torch.cat([torch.zeros_like(ref_latents), ref_latents]) if self.do_classifier_free_guidance else ref_latents
        self.unet_ref.to(device=device)
        cached_res = self.unet_ref(
            latent_model_input.unsqueeze(2),
            0,
            encoder_hidden_states=prompt_embeds_ref,
            return_dict=False,
        )[1]
        self.unet_ref.cpu()

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=prompt_embeds.dtype)

        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        base_noise = randn_tensor([batch_size, c, h, w], dtype=prompt_embeds.dtype, generator=generator).to(device=device)
        latents = base_noise * scheduler.init_noise_sigma
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        self._num_timesteps = len(timesteps)
        self.unet.to(device=device)
        pbar = comfy.utils.ProgressBar(num_inference_steps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input.unsqueeze(2),
                    t,
                    encoder_hidden_states=prompt_embeds,
                    reference_hidden_states=cached_res,
                    control_hidden_states=control_latents,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0][:,:,0,:,:]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()
                    pbar.update(1)
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(scheduler, "order", 1)
                        callback(step_idx, t, latents)

        self.unet.cpu()

        self.vae_decode.to(device=device)
        if not output_type == "latent":
            image = self.vae_decode.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
        else:
            image = latents
        self.vae_decode.cpu()

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None), latents.detach().cpu() / self.vae.config.scaling_factor
