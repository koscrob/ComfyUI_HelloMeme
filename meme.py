import json
import os
import os.path as osp
import random
import sys
import subprocess
import importlib.metadata

cur_dir = osp.dirname(osp.abspath(__file__))

installed_packages = [package.name for package in importlib.metadata.distributions()]

REQUIRED = {
    'diffusers':'0.31.0', 'transformers':'4.46.3', 'einops':'0.8.0', 'opencv-python':'4.10.0.84', 'tqdm':'4.67.0',
    'pillow':'10.2.0', 'onnxruntime-gpu':'1.18.1', 'onnx':'1.17.0', 'safetensors':'0.4.5',
    'accelerate':'1.1.1', 'peft':'0.13.2'
}

missing = [name for name in REQUIRED.keys() if name not in installed_packages]
missing_params = ' '.join([f'{k}=={REQUIRED[k]}' for k in missing])
print("missing pkgs", missing_params)

if missing:
    python = sys.executable
    os.system(f'{sys.executable} -m pip install {missing_params}')
    # subprocess.check_call([python, '-m', 'pip', 'install', missing_params], stdout=subprocess.DEVNULL)

import numpy as np
import cv2

from PIL import Image

import torch
from einops import rearrange

import folder_paths


from .hellomeme.utils import (get_drive_expression,
                              get_drive_expression_pd_fgc,
                              gen_control_heatmaps,
                              get_drive_pose,
                              crop_and_resize,
                              det_landmarks,
                              get_torch_device,
                              append_pipline_weights,
                              load_face_toolkits
                              )
from .hellomeme import HMImagePipeline, HMVideoPipeline, HM3ImagePipeline, HM3VideoPipeline

config_path = osp.join(cur_dir, 'hellomeme', 'model_config.json')
with open(config_path, 'r') as f:
    MODEL_CONFIG = json.load(f)

DEFAULT_PROMPT = MODEL_CONFIG['prompt']

def get_models_files():
    checkpoint_files = folder_paths.get_filename_list("checkpoints")
    checkpoint_files = list(MODEL_CONFIG['sd15']['checkpoints'].keys()) + checkpoint_files

    vae_files = folder_paths.get_filename_list("vae")
    vae_files = ["[vae] " + x for x in vae_files] + \
                ["[checkpoint] " + x for x in checkpoint_files]
    vae_files = ['same as checkpoint', 'SD1.5 default vae'] + vae_files

    lora_files = folder_paths.get_filename_list("loras")
    lora_files = ['None'] + list(MODEL_CONFIG['sd15']['loras'].keys()) + lora_files

    return checkpoint_files, vae_files, lora_files

def format_model_path(pipeline, config, checkpoint, vae, lora, stylize, lora_scale, deployment):
    if checkpoint and not checkpoint.startswith('SD1.5'):
        if checkpoint in config['sd15']['checkpoints']:
            tmp_checkpoint_name = config['sd15']['checkpoints'][checkpoint]
            if deployment == 'modelscope':
                from modelscope import snapshot_download
                checkpoint_path = snapshot_download(tmp_checkpoint_name)
            else:
                checkpoint_path = tmp_checkpoint_name
        else:
            checkpoint_path = folder_paths.get_full_path_or_raise("checkpoints", checkpoint)
    else:
        checkpoint_path = checkpoint

    if vae and vae.startswith("[checkpoint] "):
        vae_path = folder_paths.get_full_path_or_raise("checkpoints", vae.replace("[checkpoint] ", ""))
    elif vae and vae.startswith("[vae] "):
        vae_path = folder_paths.get_full_path_or_raise("vae", vae.replace("[vae] ", ""))
    else:
        vae_path = vae

    if lora and not lora.startswith('None'):
        if lora in config['sd15']['loras']:
            tmp_lora_info = config['sd15']['loras'][lora]
            if deployment == 'modelscope':
                from modelscope import snapshot_download
                lora_path = osp.join(snapshot_download(tmp_lora_info[0]), tmp_lora_info[1])
            else:
                from huggingface_hub import hf_hub_download
                lora_path = hf_hub_download(tmp_lora_info[0], filename=tmp_lora_info[1])
        else:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora)
    else:
        lora_path = lora

    append_pipline_weights(pipeline, checkpoint_path=checkpoint_path, lora_path=lora_path, vae_path=vae_path,
                           stylize=stylize, lora_scale=lora_scale)


class HMImagePipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        checkpoint_files, vae_files, lora_files = get_models_files()

        return {
            "optional": {
                "checkpoint": (checkpoint_files, ),
                "lora": (lora_files, ),
                "vae": (vae_files, ),
                "version": (['v1', 'v2', 'v3'], ),
                "stylize": (['x1', 'x2'], ),
                "deployment": (['huggingface', 'modelscope'], ),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("HMIMAGEPIPELINE", )
    RETURN_NAMES = ("hm_image_pipeline", )
    FUNCTION = "load_pipeline"
    CATEGORY = "hellomeme"
    def load_pipeline(self, checkpoint=None, lora=None, vae=None,
                      version='v2', stylize='x1', deployment='huggingface', lora_scale=1.0):
        dtype = torch.float16
        if deployment == 'modelscope':
            from modelscope import snapshot_download
            sd1_5_dir = snapshot_download('songkey/stable-diffusion-v1-5')
        else:
            sd1_5_dir = "songkey/stable-diffusion-v1-5"
        if version == 'v3':
            pipeline = HM3ImagePipeline.from_pretrained(sd1_5_dir)
        else:
            pipeline = HMImagePipeline.from_pretrained(sd1_5_dir)
        pipeline.to(dtype=dtype)
        pipeline.caryomitosis(version=version, modelscope=deployment=='modelscope')

        format_model_path(pipeline, MODEL_CONFIG, checkpoint, vae, lora, stylize, lora_scale, deployment)

        pipeline.insert_hm_modules(version=version, dtype=dtype, modelscope=deployment=='modelscope')
        
        return (pipeline, )


class HMVideoPipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        checkpoint_files, vae_files, lora_files = get_models_files()

        return {
            "optional": {
                "checkpoint": (checkpoint_files, ),
                "lora": (lora_files, ),
                "vae": (vae_files, ),
                "version": (['v1', 'v2', 'v3'], ),
                "stylize": (['x1', 'x2'], ),
                "deployment": (['huggingface', 'modelscope'], ),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("HMVIDEOPIPELINE",)
    RETURN_NAMES = ("hm_video_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "hellomeme"

    def load_pipeline(self, checkpoint=None, lora=None, vae=None,
                      version='v2', stylize='x1', deployment='huggingface', lora_scale=1.0):
        dtype = torch.float16
        if deployment == 'modelscope':
            from modelscope import snapshot_download
            sd1_5_dir = snapshot_download('songkey/stable-diffusion-v1-5')
        else:
            sd1_5_dir = "songkey/stable-diffusion-v1-5"

        if version == 'v3':
            pipeline = HM3VideoPipeline.from_pretrained(sd1_5_dir)
        else:
            pipeline = HMVideoPipeline.from_pretrained(sd1_5_dir)
        pipeline.to(dtype=dtype)
        pipeline.caryomitosis(version=version, modelscope=deployment=='modelscope')

        format_model_path(pipeline, MODEL_CONFIG, checkpoint, vae, lora, stylize, lora_scale, deployment)

        pipeline.insert_hm_modules(version=version, dtype=dtype, modelscope=deployment=='modelscope')

        return (pipeline,)


class HMFaceToolkitsLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "gpu_id": ("INT", {"default": 0, "min": -1, "max": 16}, ),
                "deployment": (['huggingface', 'modelscope'], ),
            }
        }

    RETURN_TYPES = ("FACE_TOOLKITS",)
    RETURN_NAMES = ("face_toolkits",)
    FUNCTION = "load_face_toolkits"
    CATEGORY = "hellomeme"

    def load_face_toolkits(self, gpu_id, deployment='huggingface'):
        dtype = torch.float16
        face_toolkits = load_face_toolkits(dtype=dtype, gpu_id=gpu_id, modelscope=deployment=='modelscope')
        return (face_toolkits, )


class CropPortrait:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "face_toolkits": ("FACE_TOOLKITS",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_portrait"
    CATEGORY = "hellomeme"

    def crop_portrait(self, image, face_toolkits):
        image_np = cv2.cvtColor((image[0] * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)

        if min(image_np.shape[:2]) < 512:
            raise Exception(f'Image size is too small -> min{image_np.shape[:2]} < 512')

        face_toolkits['face_aligner'].reset_track()
        faces = face_toolkits['face_aligner'].forward(image_np)
        if len(faces) > 0:
            face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                    x['face_rect'][3] - x['face_rect'][1]))[-1]
            ref_landmark = face['pre_kpt_222']

            new_image, new_landmark = crop_and_resize(image_np[np.newaxis, :,:,:], ref_landmark[np.newaxis, :,:], 512, crop=True)
            # for x, y in new_landmark[0]:
            #     cv2.circle(new_image[0], (int(x), int(y)), 2, (0, 255, 0), -1)
        else:
            raise Exception('No face detected')
        new_image = cv2.cvtColor(new_image[0], cv2.COLOR_RGB2BGR)
        return (torch.from_numpy(new_image[np.newaxis, :,:,:]).float() / 255., )


class GetFaceLandmarks:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FACELANDMARKS222",)
    RETURN_NAMES = ("landmarks",)
    FUNCTION = "get_face_landmarks"
    CATEGORY = "hellomeme"

    def get_face_landmarks(self, face_toolkits, images):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        frame_num = len(frame_list)
        if frame_num == 0:
            raise Exception('No image detected')
        _, landmark_list = det_landmarks(face_toolkits['face_aligner'], frame_list)
        if len(frame_list) != frame_num:
            raise Exception('Not all images have face detected!')

        return (torch.from_numpy(landmark_list).float(), )


class GetDrivePose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
                "landmarks": ("FACELANDMARKS222",),
            }
        }
    RETURN_TYPES = ("DRIVE_POSE",)
    RETURN_NAMES = ("drive_pose",)
    FUNCTION = "get_drive_pose"
    CATEGORY = "hellomeme"
    def get_drive_pose(self, face_toolkits, images, landmarks):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        landmarks = landmarks.cpu().numpy()

        rot_list, trans_list = get_drive_pose(face_toolkits, frame_list, landmarks, save_size=512)
        return (dict(rot=np.stack(rot_list), trans=np.stack(trans_list)), )


class GetDriveExpression:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
                "landmarks": ("FACELANDMARKS222",),
            }
        }
    RETURN_TYPES = ("DRIVE_EXPRESSION",)
    RETURN_NAMES = ("drive_exp",)
    FUNCTION = "get_drive_expression"
    CATEGORY = "hellomeme"
    def get_drive_expression(self, face_toolkits, images, landmarks):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        landmarks = landmarks.cpu().numpy()
        exp_dict = get_drive_expression(face_toolkits, frame_list, landmarks)
        return (exp_dict, )


class GetDriveExpression2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
                "landmarks": ("FACELANDMARKS222",),
            }
        }
    RETURN_TYPES = ("DRIVE_EXPRESSION2",)
    RETURN_NAMES = ("drive_exp2",)
    FUNCTION = "get_drive_expression"
    CATEGORY = "hellomeme"
    def get_drive_expression(self, face_toolkits, images, landmarks):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        landmarks = landmarks.cpu().numpy()
        exp_dict = get_drive_expression_pd_fgc(face_toolkits, frame_list, landmarks)
        return (exp_dict, )


class HMPipelineImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hm_image_pipeline": ("HMIMAGEPIPELINE",),
                "face_toolkits": ("FACE_TOOLKITS",),
                "ref_image": ("IMAGE",),
                "drive_pose": ("DRIVE_POSE",),
                "trans_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "prompt": ("STRING", {"default": DEFAULT_PROMPT}),
                "negative_prompt": ("STRING", {"default": ''}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "gpu_id": ("INT", {"default": 0, "min": -1, "max": 16}, ),
            },
                "optional": {
                    "drive_exp": ("DRIVE_EXPRESSION", {"default": None},),
                    "drive_exp2": ("DRIVE_EXPRESSION2", {"default": None},),
                }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", )
    FUNCTION = "sample"
    CATEGORY = "hellomeme"

    def sample(self,
               hm_image_pipeline,
               face_toolkits,
               ref_image,
               drive_pose,
               drive_exp=None,
               drive_exp2=None,
               trans_ratio='0.0',
               prompt=DEFAULT_PROMPT,
               negative_prompt='',
               steps=25,
               seed=0,
               guidance_scale=2.0,
               gpu_id=0
               ):
        device = get_torch_device(gpu_id)

        image_np = (ref_image[0] * 255).cpu().numpy().astype(np.uint8)
        if min(image_np.shape[:2]) < 512:
            raise Exception(f'Reference image size is too small -> min{image_np.shape[:2]} < 512')

        image_np = cv2.resize(image_np, (512, 512))
        image_pil = Image.fromarray(image_np)

        face_toolkits['face_aligner'].reset_track()
        faces = face_toolkits['face_aligner'].forward(image_np)
        if len(faces) == 0: raise Exception('No face detected')

        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']

        _, ref_trans = face_toolkits['h3dmm'].forward_params(image_np, ref_landmark)

        drive_rot, drive_trans = drive_pose['rot'], drive_pose['trans']
        condition = gen_control_heatmaps(drive_rot, drive_trans, ref_trans, 512, trans_ratio)
        drive_params = dict(condition=condition.unsqueeze(0).to(dtype=torch.float16, device='cpu'))

        if isinstance(drive_exp, dict):
            drive_params.update(drive_exp)
        if isinstance(drive_exp2, dict):
            drive_params.update(drive_exp2)

        generator = torch.Generator().manual_seed(seed)

        result_img, latents = hm_image_pipeline(
            prompt=[prompt],
            strength=1.0,
            image=image_pil,
            drive_params=drive_params,
            num_inference_steps=steps,
            negative_prompt=[negative_prompt],
            guidance_scale=guidance_scale,
            generator=generator,
            device=device,
            output_type='np'
        )
        return (torch.from_numpy(np.clip(result_img[0], 0, 1)), dict(samples=latents), )


class HMPipelineVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
                    "required":{
                        "hm_video_pipeline": ("HMVIDEOPIPELINE",),
                        "face_toolkits": ("FACE_TOOLKITS",),
                        "ref_image": ("IMAGE",),
                        "drive_pose": ("DRIVE_POSE",),
                        "trans_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                        "patch_overlap": ("INT", {"default": 4, "min": 0, "max": 5}),
                        "prompt": ("STRING", {"default": DEFAULT_PROMPT}),
                        "negative_prompt": ("STRING", {"default": ''}),
                        "steps": ("INT", {"default": 25, "min": 1, "max": 10000,
                                          "tooltip": "The number of steps used in the denoising process."}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                         "tooltip": "The random seed used for creating the noise."}),
                        "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                        "gpu_id": ("INT", {"default": 0, "min": -1, "max": 16}, ),
                     },
                    "optional": {
                        "drive_exp": ("DRIVE_EXPRESSION", {"default": None},),
                        "drive_exp2": ("DRIVE_EXPRESSION2", {"default": None},),
                    }
                }

    RETURN_TYPES = ("IMAGE", "LATENT", )
    FUNCTION = "sample"
    CATEGORY = "hellomeme"

    def sample(self,
                hm_video_pipeline,
                face_toolkits,
                ref_image,
                drive_pose,
                drive_exp=None,
                drive_exp2=None,
                trans_ratio=0.0,
                patch_overlap=4,
                prompt=DEFAULT_PROMPT,
                negative_prompt="",
                steps=25,
                seed=0,
                guidance_scale=2.0,
                gpu_id=0
        ):
        device = get_torch_device(gpu_id)

        image_np = (ref_image[0] * 255).cpu().numpy().astype(np.uint8)
        if min(image_np.shape[:2]) < 512:
            raise Exception(f'Reference image size is too small -> min{image_np.shape[:2]} < 512')

        image_np = cv2.resize(image_np, (512, 512))
        image_pil = Image.fromarray(image_np)

        face_toolkits['face_aligner'].reset_track()
        faces = face_toolkits['face_aligner'].forward(image_np)
        face_toolkits['face_aligner'].reset_track()
        if len(faces) == 0: raise Exception('No face detected')

        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']

        _, ref_trans = face_toolkits['h3dmm'].forward_params(image_np, ref_landmark)

        generator = torch.Generator().manual_seed(seed)

        drive_rot, drive_trans = drive_pose['rot'], drive_pose['trans']
        condition = gen_control_heatmaps(drive_rot, drive_trans, ref_trans, 512, trans_ratio)
        drive_params = dict(condition=condition.unsqueeze(0).to(dtype=torch.float16, device='cpu'))

        if isinstance(drive_exp, dict):
            drive_params.update(drive_exp)
        if isinstance(drive_exp2, dict):
            drive_params.update(drive_exp2)

        res_frames, latents = hm_video_pipeline(
            prompt=[prompt],
            strength=1.0,
            image=image_pil,
            chunk_overlap=patch_overlap,
            drive_params=drive_params,
            num_inference_steps=steps,
            negative_prompt=[negative_prompt],
            guidance_scale=guidance_scale,
            generator=generator,
            device=device,
            output_type='np'
        )
        res_frames = [np.clip(x[0], 0, 1) for x in res_frames]
        latents = rearrange(latents[0], 'c f h w -> f c h w')

        return (torch.from_numpy(np.array(res_frames)), dict(samples=latents), )


NODE_CLASS_MAPPINGS = {
    "HMImagePipelineLoader": HMImagePipelineLoader,
    "HMVideoPipelineLoader": HMVideoPipelineLoader,
    "HMFaceToolkitsLoader": HMFaceToolkitsLoader,
    "HMPipelineImage": HMPipelineImage,
    "HMPipelineVideo": HMPipelineVideo,
    "CropPortrait": CropPortrait,
    "GetFaceLandmarks": GetFaceLandmarks,
    "GetDrivePose": GetDrivePose,
    "GetDriveExpression": GetDriveExpression,
    "GetDriveExpression2": GetDriveExpression2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HMImagePipelineLoader": "Load HelloMemeImage Pipeline",
    "HMVideoPipelineLoader": "Load HelloMemeVideo Pipeline",
    "HMFaceToolkitsLoader": "Load Face Toolkits",
    "HMPipelineImage": "HelloMeme Image Pipeline",
    "HMPipelineVideo": "HelloMeme Video Pipeline",
    "CropPortrait": "Crop Portrait",
    "GetFaceLandmarks": "Get Face Landmarks",
    "GetDrivePose": "Get Drive Pose",
    "GetDriveExpression": "Get Drive Expression",
    "GetDriveExpression2": "Get Drive Expression V2",
}
