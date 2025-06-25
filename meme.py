import json
import os
import os.path as osp
import sys
import importlib.metadata

cur_dir = osp.dirname(osp.abspath(__file__))

installed_packages = [package.name for package in importlib.metadata.distributions()]

REQUIRED = {
    'diffusers':'0.33.1', 'transformers':'4.46.3', 'einops':'0.8.0', 'opencv-python':'4.10.0.84', 'tqdm':'4.67.0',
    'pillow':'10.2.0', 'onnxruntime-gpu':'1.18.1', 'onnx':'1.17.0', 'safetensors':'0.4.5',
    'accelerate':'1.1.1', 'peft':'0.13.2', 'modelscope':'', 'huggingface-hub':''
}

missing = [name for name in REQUIRED.keys() if name not in installed_packages]
missing_params = ' '.join([f'{k}' for k in missing])
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
                              det_landmarks,
                              get_torch_device,
                              append_pipline_weights,
                              load_face_toolkits
                              )
from .hellomeme import (HMImagePipeline, HMVideoPipeline,
                        HM3ImagePipeline, HM3VideoPipeline,
                        HM5ImagePipeline, HM5VideoPipeline,
                        download_file_from_cloud,
                        creat_model_from_cloud)

config_path = osp.join(cur_dir, 'hellomeme', 'model_config.json')
with open(config_path, 'r') as f:
    MODEL_CONFIG = json.load(f)

DEFAULT_PROMPT = MODEL_CONFIG['prompt']
DEFAULT_PROMPT_NEW = MODEL_CONFIG['prompt_new']

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
            checkpoint_path = config['sd15']['checkpoints'][checkpoint]
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
            lora_path = download_file_from_cloud(tmp_lora_info[0], tmp_lora_info[1], modelscope=deployment == 'modelscope')
        else:
            lora_path = folder_paths.get_full_path_or_raise("loras", lora)
    else:
        lora_path = lora

    append_pipline_weights(pipeline, checkpoint_path=checkpoint_path, lora_path=lora_path, vae_path=vae_path,
                           stylize=stylize, lora_scale=lora_scale, modelscope=deployment == 'modelscope')


class HMImagePipelineLoader:
    @classmethod
    def INPUT_TYPES(s):
        checkpoint_files, vae_files, lora_files = get_models_files()

        return {
            "optional": {
                "checkpoint": (checkpoint_files, ),
                "lora": (lora_files, ),
                "vae": (vae_files, ),
                "version": (['v5b', 'v5', 'v4', 'v3', 'v2', 'v1'], ),
                "stylize": (['x1', 'x2'], ),
                "deployment": (['huggingface', 'modelscope'], ),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "dtype": (['fp32', 'fp16'], ),
            }
        }
    RETURN_TYPES = ("HMIMAGEPIPELINE", )
    RETURN_NAMES = ("hm_image_pipeline", )
    FUNCTION = "load_pipeline"
    CATEGORY = "hellomeme"
    def load_pipeline(self, checkpoint=None, lora=None, vae=None,
                      version='v2', stylize='x1', deployment='huggingface', lora_scale=1.0, dtype='fp32'):
        dtype = torch.float32 if dtype == 'fp32' else torch.float16

        if version == 'v3' or version == 'v4':
            pipeline = creat_model_from_cloud(HM3ImagePipeline, "songkey/stable-diffusion-v1-5",
                                              modelscope=deployment == 'modelscope')
        elif version == 'v5' or version == 'v5b':
            pipeline = creat_model_from_cloud(HM5ImagePipeline, "songkey/stable-diffusion-v1-5",
                                              modelscope=deployment == 'modelscope')
        else:
            pipeline = creat_model_from_cloud(HMImagePipeline, "songkey/stable-diffusion-v1-5",
                                              modelscope=deployment == 'modelscope')
        pipeline.to(dtype=dtype)
        pipeline.caryomitosis(version=version, modelscope=deployment == 'modelscope')

        format_model_path(pipeline, MODEL_CONFIG, checkpoint, vae, lora, stylize, lora_scale, deployment)

        pipeline.insert_hm_modules(version=version, dtype=dtype, modelscope=deployment == 'modelscope')
        
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
                "version": (['v5', 'v4', 'v3', 'v2', 'v1'], ),
                "stylize": (['x1', 'x2'], ),
                "deployment": (['huggingface', 'modelscope'], ),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "dtype": (['fp32', 'fp16'], ),
            }
        }

    RETURN_TYPES = ("HMVIDEOPIPELINE",)
    RETURN_NAMES = ("hm_video_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "hellomeme"

    def load_pipeline(self, checkpoint=None, lora=None, vae=None,
                      version='v2', stylize='x1', deployment='huggingface', lora_scale=1.0, dtype='fp32'):
        dtype = torch.float32 if dtype == 'fp32' else torch.float16

        if version == 'v3' or version == 'v4':
            pipeline = creat_model_from_cloud(HM3VideoPipeline, "songkey/stable-diffusion-v1-5",
                                              modelscope=deployment == 'modelscope')
        elif version == 'v5' or version == 'v5b':
            pipeline = creat_model_from_cloud(HM5VideoPipeline, "songkey/stable-diffusion-v1-5",
                                              modelscope=deployment == 'modelscope')
        else:
            pipeline = creat_model_from_cloud(HMVideoPipeline, "songkey/stable-diffusion-v1-5",
                                              modelscope=deployment == 'modelscope')
        pipeline.to(dtype=dtype)
        pipeline.caryomitosis(version=version, modelscope=deployment == 'modelscope')

        format_model_path(pipeline, MODEL_CONFIG, checkpoint, vae, lora, stylize, lora_scale, deployment)

        pipeline.insert_hm_modules(version=version, dtype=dtype, modelscope=deployment == 'modelscope')

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
        dtype = torch.float32
        face_toolkits = load_face_toolkits(dtype=dtype, gpu_id=gpu_id, modelscope=deployment=='modelscope')
        return (face_toolkits, )


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
        assert frame_num > 0, 'No image detected'
        _, landmark_list = det_landmarks(face_toolkits['face_aligner'], frame_list)
        assert len(frame_list) == frame_num, 'Not all images have face detected!'

        return (torch.from_numpy(landmark_list).float(), )


class GetHeadPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
                "landmarks": ("FACELANDMARKS222",),
                "crop": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
            }
        }
    RETURN_TYPES = ("HEAD_POSE",)
    RETURN_NAMES = ("head_pose",)
    FUNCTION = "get_head_pose"
    CATEGORY = "hellomeme"
    def get_head_pose(self, face_toolkits, images, landmarks, crop):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        landmarks = landmarks.cpu().numpy()

        new_frames, new_landmarks, rot_list, trans_list = get_drive_pose(face_toolkits, frame_list, landmarks, save_size=512, crop=crop)
        return (dict(rot=np.stack(rot_list), trans=np.stack(trans_list), frame=np.stack(new_frames), landmarks=np.stack(new_landmarks)), )


class GetExpression:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
                "landmarks": ("FACELANDMARKS222",),
            }
        }
    RETURN_TYPES = ("EXPRESSION",)
    RETURN_NAMES = ("expression",)
    FUNCTION = "get_expression"
    CATEGORY = "hellomeme"
    def get_expression(self, face_toolkits, images, landmarks):
        frame_list = [cv2.cvtColor((frame * 255).cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2RGB) for frame in images]
        landmarks = landmarks.cpu().numpy()
        exp_dict = get_drive_expression(face_toolkits, frame_list, landmarks)
        return (exp_dict, )


class GetExpression2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_toolkits": ("FACE_TOOLKITS",),
                "images": ("IMAGE",),
                "landmarks": ("FACELANDMARKS222",),
            }
        }
    RETURN_TYPES = ("EXPRESSION",)
    RETURN_NAMES = ("expression",)
    FUNCTION = "get_expression"
    CATEGORY = "hellomeme"
    def get_expression(self, face_toolkits, images, landmarks):
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
                "ref_head_pose": ("HEAD_POSE",),
                "ref_expression": ("EXPRESSION",),
                "drive_head_pose": ("HEAD_POSE",),
                "drive_expression": ("EXPRESSION",),
                "trans_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "prompt": ("STRING", {"default": ''}),
                "negative_prompt": ("STRING", {"default": ''}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 10000,
                                  "tooltip": "The number of steps used in the denoising process."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
                "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "gpu_id": ("INT", {"default": 0, "min": -1, "max": 16}, ),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", )
    FUNCTION = "sample"
    CATEGORY = "hellomeme"

    def sample(self,
                hm_image_pipeline,
                ref_head_pose,
                ref_expression,
                drive_head_pose,
                drive_expression,
                trans_ratio='0.0',
                prompt='',
                negative_prompt='',
                steps=25,
                seed=0,
                guidance_scale=2.0,
                gpu_id=0
               ):
        device = get_torch_device(gpu_id)
        dtype = hm_image_pipeline.dtype

        PROMPT = DEFAULT_PROMPT_NEW if hm_image_pipeline.version == 'v5b' else DEFAULT_PROMPT
        prompt = PROMPT if prompt == '' else prompt + ", " + PROMPT

        image_np = cv2.cvtColor(ref_head_pose['frame'][0], cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_np)

        ref_trans = ref_head_pose['trans'][0]

        drive_rot, drive_trans = drive_head_pose['rot'], drive_head_pose['trans']
        condition = gen_control_heatmaps(drive_rot, drive_trans, ref_trans, 512, trans_ratio)
        drive_params = dict(condition=condition.unsqueeze(0))
        drive_params.update(drive_expression)
        for k, v in drive_params.items():
            drive_params[k] = v.to(dtype=dtype, device='cpu')

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
                        "ref_head_pose": ("HEAD_POSE",),
                        "ref_expression": ("EXPRESSION",),
                        "drive_head_pose": ("HEAD_POSE",),
                        "drive_expression": ("EXPRESSION",),
                        "trans_ratio": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                        "patch_overlap": ("INT", {"default": 4, "min": 0, "max": 5}),
                        "prompt": ("STRING", {"default": ''}),
                        "negative_prompt": ("STRING", {"default": ''}),
                        "steps": ("INT", {"default": 25, "min": 1, "max": 10000,
                                          "tooltip": "The number of steps used in the denoising process."}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                         "tooltip": "The random seed used for creating the noise."}),
                        "guidance_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                        "gpu_id": ("INT", {"default": 0, "min": -1, "max": 16}, ),
                     },
                }

    RETURN_TYPES = ("IMAGE", "LATENT", )
    FUNCTION = "sample"
    CATEGORY = "hellomeme"

    def sample(self,
                hm_video_pipeline,
                ref_head_pose,
                ref_expression,
                drive_head_pose,
                drive_expression,
                trans_ratio=0.0,
                patch_overlap=4,
                prompt='',
                negative_prompt="",
                steps=25,
                seed=0,
                guidance_scale=2.0,
                gpu_id=0
        ):
        device = get_torch_device(gpu_id)
        dtype = hm_video_pipeline.dtype

        prompt = DEFAULT_PROMPT if prompt == '' else prompt + ", " + DEFAULT_PROMPT

        image_np = cv2.cvtColor(ref_head_pose['frame'][0], cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_np)

        ref_rot, ref_trans = ref_head_pose['rot'], ref_head_pose['trans']
        generator = torch.Generator().manual_seed(seed)

        drive_rot, drive_trans = drive_head_pose['rot'], drive_head_pose['trans']
        condition = gen_control_heatmaps(drive_rot, drive_trans, ref_trans[0], 512, trans_ratio)
        ref_condition = gen_control_heatmaps(ref_rot, ref_trans, ref_trans[0], 512, 0.0)
        drive_params = dict(condition=condition.unsqueeze(0))
        ref_params = dict(condition=ref_condition.unsqueeze(0))
        drive_params.update(drive_expression)
        ref_params.update(ref_expression)

        for k, v in drive_params.items():
            drive_params[k] = v.to(dtype=dtype)
        for k, v in ref_params.items():
            ref_params[k] = v.to(dtype=dtype, device='cpu')

        res_frames, latents = hm_video_pipeline(
            prompt=[prompt],
            strength=1.0,
            image=image_pil,
            chunk_overlap=patch_overlap,
            ref_params=ref_params,
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
    "GetFaceLandmarks": GetFaceLandmarks,
    "GetHeadPose": GetHeadPose,
    "GetExpression": GetExpression,
    "GetExpression2": GetExpression2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HMImagePipelineLoader": "Load HelloMemeImage Pipeline",
    "HMVideoPipelineLoader": "Load HelloMemeVideo Pipeline",
    "HMFaceToolkitsLoader": "Load Face Toolkits",
    "HMPipelineImage": "HelloMeme Image Pipeline",
    "HMPipelineVideo": "HelloMeme Video Pipeline",
    "GetFaceLandmarks": "Get Face Landmarks",
    "GetHeadPose": "Get Head Pose",
    "GetExpression": "Get Face Expression",
    "GetExpression2": "Get Face Expression V2",
}
