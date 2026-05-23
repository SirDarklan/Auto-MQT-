from __future__ import annotations

import contextlib
import io
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    visual_tokens: int
    latency_s: float


class MqtLlavaBackend(Protocol):
    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        visual_tokens: int,
        max_new_tokens: int = 64,
    ) -> str:
        ...


class PlaceholderMqtLlavaBackend:
    """Replace this with the actual MQT-LLaVA repo/checkpoint inference call."""

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        visual_tokens: int,
        max_new_tokens: int = 64,
    ) -> str:
        raise NotImplementedError(
            "MQT-LLaVA is not configured. Set MQT_LLAVA_REPO to your cloned "
            "MQT-LLaVA path, or install the repo in the active Python environment. "
            "Example: MQT_LLAVA_REPO=/path/to/MQT-LLaVA python3 src/evaluate_token_policy.py ..."
        )


class MqtLlavaEvalBackend:
    """Backend that calls the public MQT-LLaVA `llava.eval.run_llava.eval_model` entry point."""

    def __init__(
        self,
        repo_path: str | Path | None = None,
        model_path: str = "gordonhu/MQT-LLaVA-7b",
        model_base: str | None = None,
        conv_mode: str = "llava_v1",
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams

        if self.repo_path is not None:
            if not (self.repo_path / "llava").exists():
                raise FileNotFoundError(f"Expected an MQT-LLaVA clone with llava/ at {self.repo_path}")
            sys.path.insert(0, str(self.repo_path))

        from llava.eval.run_llava import eval_model
        from llava.mm_utils import get_model_name_from_path

        self._eval_model = eval_model
        self._get_model_name_from_path = get_model_name_from_path

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        visual_tokens: int,
        max_new_tokens: int = 64,
    ) -> str:
        args = type(
            "Args",
            (),
            {
                "model_path": self.model_path,
                "model_base": self.model_base,
                "model_name": self._get_model_name_from_path(self.model_path),
                "query": prompt,
                "conv_mode": self.conv_mode,
                "num_visual_tokens": int(visual_tokens),
                "image_file": str(image_path),
                "sep": ",",
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_beams": self.num_beams,
                "max_new_tokens": int(max_new_tokens),
            },
        )()

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            returned = self._eval_model(args)
        if isinstance(returned, str) and returned.strip():
            return returned.strip()

        printed = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
        if not printed:
            return ""
        return printed[-1]


class MqtLlavaPersistentBackend:
    """Persistent MQT-LLaVA backend for multi-example oracle labeling."""

    def __init__(
        self,
        repo_path: str | Path | None = None,
        model_path: str = "gordonhu/MQT-LLaVA-7b",
        model_base: str | None = None,
        conv_mode: str | None = None,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
        device_map: str = "auto",
        offload_folder: str | Path = "offload",
        load_8bit: bool = False,
        load_4bit: bool = False,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.device_map = device_map
        self.offload_folder = Path(offload_folder)
        self.load_8bit = bool(load_8bit)
        self.load_4bit = bool(load_4bit)
        self.offload_folder.mkdir(parents=True, exist_ok=True)

        if self.repo_path is not None:
            if not (self.repo_path / "llava").exists():
                raise FileNotFoundError(f"Expected an MQT-LLaVA clone with llava/ at {self.repo_path}")
            sys.path.insert(0, str(self.repo_path))

        import torch
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_PLACEHOLDER,
            IMAGE_TOKEN_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.eval.run_llava import load_images
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from transformers.utils import logging as hf_logging

        self.torch = torch
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.IMAGE_PLACEHOLDER = IMAGE_PLACEHOLDER
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.conv_templates = conv_templates
        self.load_images = load_images
        self.process_images = process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.model_name = get_model_name_from_path(model_path)

        disable_torch_init()
        load_kwargs = {
            "device_map": device_map,
            "offload_folder": str(self.offload_folder),
            "offload_state_dict": True,
            "load_8bit": self.load_8bit,
            "load_4bit": self.load_4bit,
        }
        quiet_load = os.environ.get("AUTO_MQT_SUPPRESS_LOAD_WARNINGS", "1") == "1"
        previous_verbosity = hf_logging.get_verbosity()
        if quiet_load:
            hf_logging.set_verbosity_error()
        try:
            try:
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    model_path,
                    model_base,
                    self.model_name,
                    **load_kwargs,
                )
            except TypeError:
                # Fallback for older llava forks that don't accept load_8bit/load_4bit.
                load_kwargs.pop("load_8bit", None)
                load_kwargs.pop("load_4bit", None)
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    model_path,
                    model_base,
                    self.model_name,
                    **load_kwargs,
                )
        finally:
            if quiet_load:
                hf_logging.set_verbosity(previous_verbosity)

    def _infer_conv_mode(self) -> str:
        if "llama-2" in self.model_name.lower():
            return "llava_llama_2"
        if "mistral" in self.model_name.lower():
            return "mistral_instruct"
        if "v1.6-34b" in self.model_name.lower():
            return "chatml_direct"
        if "v1" in self.model_name.lower():
            return "llava_v1"
        if "mpt" in self.model_name.lower():
            return "mpt"
        return "llava_v0"

    def _build_prompt(self, query: str) -> str:
        qs = query
        image_token_se = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN
        if self.IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(self.IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(self.IMAGE_PLACEHOLDER, self.DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = self.DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_mode = self.conv_mode or self._infer_conv_mode()
        conv = self.conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        visual_tokens: int,
        max_new_tokens: int = 64,
    ) -> str:
        prompt_text = self._build_prompt(prompt)
        images = self.load_images([str(image_path)])
        image_sizes = [image.size for image in images]
        images_tensor = self.process_images(
            images,
            self.image_processor,
            self.model.config,
        ).to(self.model.device, dtype=self.torch.float16)

        input_ids = self.tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self.model.device)

        with self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                num_visual_tokens=int(visual_tokens),
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=int(max_new_tokens),
                use_cache=True,
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


class BlipVqaSmokeBackend:
    """Small local VQA backend for pipeline smoke tests.

    This backend ignores `visual_tokens` and serves only as a lightweight
    end-to-end check before running expensive MQT-LLaVA jobs on Colab.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
        device: str = "auto",
        max_new_tokens: int = 20,
    ) -> None:
        import torch
        from PIL import Image
        from transformers import BlipForQuestionAnswering, BlipProcessor

        self.torch = torch
        self.Image = Image
        self.max_new_tokens = int(max_new_tokens)

        self.device = (
            torch.device("cuda")
            if device == "auto" and torch.cuda.is_available()
            else torch.device("cpu" if device == "auto" else device)
        )
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        visual_tokens: int,
        max_new_tokens: int = 64,
    ) -> str:
        with self.Image.open(image_path) as image:
            image = image.convert("RGB")
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        generation_limit = min(int(max_new_tokens), self.max_new_tokens)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=generation_limit)
        text = self.processor.decode(output[0], skip_special_tokens=True).strip()
        return text


def build_mqt_llava_backend() -> MqtLlavaBackend:
    repo_path = os.environ.get("MQT_LLAVA_REPO")
    model_path = os.environ.get("MQT_LLAVA_MODEL_PATH", "gordonhu/MQT-LLaVA-7b")
    conv_mode = os.environ.get("MQT_LLAVA_CONV_MODE")
    backend = os.environ.get("MQT_LLAVA_BACKEND", "persistent")
    device_map = os.environ.get("MQT_LLAVA_DEVICE_MAP", "auto")
    offload_folder = os.environ.get("MQT_LLAVA_OFFLOAD_FOLDER", "offload")
    load_8bit = os.environ.get("MQT_LLAVA_LOAD_8BIT", "0") == "1"
    load_4bit = os.environ.get("MQT_LLAVA_LOAD_4BIT", "0") == "1"
    smoke_model_name = os.environ.get("AUTO_MQT_SMOKE_MODEL", "Salesforce/blip-vqa-base")
    smoke_device = os.environ.get("AUTO_MQT_SMOKE_DEVICE", "auto")
    smoke_max_new_tokens = int(os.environ.get("AUTO_MQT_SMOKE_MAX_NEW_TOKENS", "20"))

    if backend == "blip_vqa_smoke":
        return BlipVqaSmokeBackend(
            model_name=smoke_model_name,
            device=smoke_device,
            max_new_tokens=smoke_max_new_tokens,
        )

    if repo_path or os.environ.get("MQT_LLAVA_USE_INSTALLED") == "1":
        if backend == "eval":
            return MqtLlavaEvalBackend(
                repo_path=repo_path,
                model_path=model_path,
                conv_mode=conv_mode or "llava_v1",
            )
        return MqtLlavaPersistentBackend(
            repo_path=repo_path,
            model_path=model_path,
            conv_mode=conv_mode,
            device_map=device_map,
            offload_folder=offload_folder,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
        )
    return PlaceholderMqtLlavaBackend()


class TimedMqtLlava:
    def __init__(self, backend: MqtLlavaBackend) -> None:
        self.backend = backend

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        visual_tokens: int,
        max_new_tokens: int = 64,
    ) -> GenerationResult:
        normalized_image_path = Path(str(image_path).replace("\\", "/"))
        start = perf_counter()
        text = self.backend.generate(
            image_path=normalized_image_path,
            prompt=prompt,
            visual_tokens=visual_tokens,
            max_new_tokens=max_new_tokens,
        )
        return GenerationResult(
            text=text,
            visual_tokens=visual_tokens,
            latency_s=perf_counter() - start,
        )
