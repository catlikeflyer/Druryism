#!/usr/bin/env python3
"""
Vision-to-Commentary Pipeline
Author: Principal AI Software Engineer

This module provides a complete, decoupled, and production-ready implementation
of a pipeline that translates sports scenes (images) into poetic, Peter Drury-style
commentary monologues. It consists of:
1. VisionAnalyzer: Local VLM-based scene analysis (using Transformers).
2. DruryCommentaryEngine: Local SLM-based poetic generation (using MLX and QLoRA adapters).
3. CommentaryPipeline: Orchestrates the stages in a single end-to-end stream.
"""

import os
import sys
import logging
from typing import Generator, Optional
from dataclasses import dataclass

# -------------------------------------------------------------------------
# Pre-flight Dependency & Environment Check
# -------------------------------------------------------------------------
REQUIRED_MODULES = ["torch", "transformers", "PIL", "mlx", "mlx_lm"]
missing_modules = []
for mod in REQUIRED_MODULES:
    try:
        __import__(mod)
    except ImportError:
        pkg_name = "Pillow" if mod == "PIL" else ("mlx-lm" if mod == "mlx_lm" else mod)
        missing_modules.append(pkg_name)

if missing_modules:
    print("\n[!] CRITICAL ERROR: Missing required dependencies to run the pipeline.", file=sys.stderr)
    print(f"[!] Please install them using:\n    pip install {' '.join(missing_modules)}\n", file=sys.stderr)
    sys.exit(1)

# Now safe to import from dependencies
import torch
from PIL import Image, ImageDraw
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DruryPipeline")


# -------------------------------------------------------------------------
# Configurations
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class VisionConfig:
    model_id: str = "Salesforce/blip-image-captioning-large"
    device: Optional[str] = None  # Autodetected if None


@dataclass(frozen=True)
class CommentaryConfig:
    base_model_id: str = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    adapter_path: str = "./drury_adapters"
    temperature: float = 0.85
    top_p: float = 0.9
    max_tokens: int = 200


# -------------------------------------------------------------------------
# 1. Vision Stage: VisionAnalyzer
# -------------------------------------------------------------------------
class VisionAnalyzer:
    """
    Handles loading the visual-language model, preprocessing local images,
    and generating factual, structural scene descriptions.
    """
    def __init__(self, config: VisionConfig = VisionConfig()):
        self.config = config
        self.device = config.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.processor: Optional[BlipProcessor] = None
        self.model: Optional[BlipForConditionalGeneration] = None
        logger.info(f"VisionAnalyzer initialized on target device: {self.device}")

    def load_model(self) -> None:
        """Loads and initializes the processor and model weights."""
        if self.model is not None and self.processor is not None:
            return

        logger.info(f"Loading vision model '{self.config.model_id}'...")
        try:
            self.processor = BlipProcessor.from_pretrained(self.config.model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.config.model_id
            ).to(self.device)
            logger.info("Vision model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise RuntimeError(f"Vision model initialization failed: {e}") from e

    def analyze_image(self, image_path: str) -> str:
        """
        Loads a local image file and generates its factual description.

        Args:
            image_path: Relative or absolute path to the local image.

        Returns:
            A string containing the factual description of the scene.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at: {image_path}")

        # Lazy loading
        self.load_model()
        
        # Verify loader loaded models correctly for type safety
        assert self.processor is not None
        assert self.model is not None

        logger.info(f"Analyzing scene from image: '{image_path}'")
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=80,
                    num_beams=3,
                    min_length=10
                )

            description = self.processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"Factual scene description generated: '{description}'")
            return description.strip()
        except Exception as e:
            logger.error(f"Error occurred during image analysis: {e}")
            raise RuntimeError(f"Image analysis failed: {e}") from e


# -------------------------------------------------------------------------
# 2. Text Stage: DruryCommentaryEngine
# -------------------------------------------------------------------------
class DruryCommentaryEngine:
    """
    Handles local MLX SLM model loading, applies fine-tuned QLoRA adapters,
    formats input scenes to ChatML format, and streams outputs.
    """
    def __init__(self, config: CommentaryConfig = CommentaryConfig()):
        self.config = config
        self.model: Optional[any] = None  # mlx model instance
        self.tokenizer: Optional[any] = None
        logger.info(f"DruryCommentaryEngine configured for model: '{config.base_model_id}'")

    def load_model(self) -> None:
        """
        Loads the MLX model. Attempts to load LoRA adapters if present;
        falls back to base model with warning if adapters folder is missing.
        """
        if self.model is not None and self.tokenizer is not None:
            return

        from mlx_lm import load

        # Check adapter directory availability
        adapter_exists = os.path.isdir(self.config.adapter_path)
        if adapter_exists:
            logger.info(f"Loading base model '{self.config.base_model_id}' with adapters from '{self.config.adapter_path}'...")
            try:
                self.model, self.tokenizer = load(
                    self.config.base_model_id,
                    adapter_path=self.config.adapter_path
                )
                logger.info("MLX model loaded with fine-tuned LoRA adapters successfully.")
                return
            except Exception as e:
                logger.warning(f"Failed to load adapters from '{self.config.adapter_path}': {e}. Falling back to base model.")

        # Fallback / Direct Base Model loading
        logger.info(f"Loading base MLX model '{self.config.base_model_id}' (no adapters)...")
        try:
            self.model, self.tokenizer = load(self.config.base_model_id)
            logger.info("Base MLX model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load MLX base model: {e}")
            raise RuntimeError(f"MLX model initialization failed: {e}") from e

    def format_prompt(self, factual_description: str) -> str:
        """
        Wraps the VLM output into the specific ChatML prompt template.

        Args:
            factual_description: Description string from Vision stage.

        Returns:
            The structured prompt string.
        """
        # Strictly enforces requested ChatML schema
        prompt = (
            "<|im_start|>system\n"
            "You are Peter Drury, the poetic football commentator. Turn the factual scene into a dramatic monologue.<|im_end|>\n"
            f"<|im_start|>user\nScene: {factual_description}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt

    def generate_commentary_stream(self, factual_description: str) -> Generator[str, None, None]:
        """
        Wraps scene description, sets sampler configurations, and streams the output.

        Args:
            factual_description: Factual scene description text.

        Yields:
            str: Token chunks of the dramatic monologue.
        """
        self.load_model()
        prompt = self.format_prompt(factual_description)
        logger.info("Starting text generation stream...")

        streamer = self._stream_generate_safe(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            temp=self.config.temperature,
            top_p=self.config.top_p
        )

        for response in streamer:
            # Handle variations between mlx_lm versions yielding strings vs objects
            if hasattr(response, "text"):
                yield response.text
            else:
                yield str(response)

    def _stream_generate_safe(self, model, tokenizer, prompt: str, max_tokens: int, temp: float, top_p: float):
        """
        Safety wrapper supporting multiple mlx_lm generations APIs (samplers vs direct parameters).
        """
        from mlx_lm import stream_generate

        # 1. Attempt sampler utility if available (preferred in latest mlx_lm versions)
        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temp, top_p=top_p)
            yield from stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)
            return
        except (ImportError, TypeError) as e:
            logger.debug(f"make_sampler initialization failed or skipped: {e}. Trying direct parameters.")

        # 2. Fallback to direct parameters (using 'temp')
        try:
            yield from stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, temp=temp, top_p=top_p)
            return
        except TypeError as e:
            logger.debug(f"Direct 'temp' generation failed: {e}. Trying direct 'temperature'.")

        # 3. Fallback to direct parameters (using 'temperature')
        try:
            yield from stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, temperature=temp, top_p=top_p)
            return
        except TypeError as e:
            logger.debug(f"Direct 'temperature' generation failed: {e}. Falling back to default generation.")

        # 4. Final fallback with default arguments
        yield from stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)


# -------------------------------------------------------------------------
# 3. Orchestrator: CommentaryPipeline
# -------------------------------------------------------------------------
class CommentaryPipeline:
    """
    Orchestrates the end-to-end execution of the pipeline, loading stages,
    handling runtime coordination, and piping visual captions to the text engine.
    """
    def __init__(self, vision_analyzer: VisionAnalyzer, commentary_engine: DruryCommentaryEngine):
        self.vision_analyzer = vision_analyzer
        self.commentary_engine = commentary_engine
        logger.info("CommentaryPipeline successfully instantiated.")

    def run(self, image_path: str) -> Generator[str, None, None]:
        """
        Executes the Vision-to-Commentary pipeline.

        Args:
            image_path: Path to the local image.

        Yields:
            str: Generated tokens of the monologue commentary.
        """
        logger.info(f"Pipeline started for image target: '{image_path}'")
        
        # Step 1: Vision Scene Factual Captioning
        try:
            factual_description = self.vision_analyzer.analyze_image(image_path)
        except Exception as e:
            logger.error(f"Pipeline crashed during the Vision stage: {e}")
            raise RuntimeError(f"Vision stage failed: {e}") from e

        # Step 2: Commentary monologue generation
        try:
            logger.info("Vision analysis complete. Initializing commentary stream orchestration...")
            stream = self.commentary_engine.generate_commentary_stream(factual_description)
            for token in stream:
                yield token
        except Exception as e:
            logger.error(f"Pipeline crashed during the Commentary Generation stage: {e}")
            raise RuntimeError(f"Commentary generation failed: {e}") from e


# -------------------------------------------------------------------------
# Mock Helpers & Out-of-the-Box Execution Logic
# -------------------------------------------------------------------------
def create_mock_sports_image(filename: str = "mock_match_pitch.jpg") -> str:
    """
    Creates a simulated sports image locally to run the script out-of-the-box
    without needing manual file setups.
    """
    if os.path.exists(filename):
        return filename

    logger.info(f"No sports image found. Creating mock soccer pitch image at '{filename}'...")
    try:
        # Green pitch background
        image = Image.new("RGB", (450, 300), color=(46, 125, 50))
        draw = ImageDraw.Draw(image)

        # Draw white pitch markings
        draw.rectangle([15, 15, 435, 285], outline=(255, 255, 255), width=3)
        draw.line([225, 15, 225, 285], fill=(255, 255, 255), width=3)
        draw.ellipse([195, 120, 255, 180], outline=(255, 255, 255), width=3)

        # Draw Goal Box
        draw.rectangle([15, 90, 75, 210], outline=(255, 255, 255), width=2)
        draw.rectangle([375, 90, 435, 210], outline=(255, 255, 255), width=2)

        # Draw a soccer ball on the pitch
        draw.ellipse([270, 135, 286, 151], fill=(255, 255, 255), outline=(0, 0, 0), width=1)

        image.save(filename)
        logger.info(f"Successfully saved mock sports image to: '{filename}'")
        return filename
    except Exception as e:
        logger.warning(f"Could not generate graphics mock image: {e}. Writing plain fallback file.")
        fallback_image = Image.new("RGB", (100, 100), color="green")
        fallback_image.save(filename)
        return filename


def find_or_create_image() -> str:
    """
    Looks for any existing image file (.jpg, .jpeg, .png) in the current directory.
    If none are found, generates a mock sports image.
    """
    valid_extensions = (".jpg", ".jpeg", ".png")
    # Sort files to ensure deterministic selection (e.g. spurs_uel.jpg before mock_match_pitch.jpg)
    for file in sorted(os.listdir(".")):
        if file.lower().endswith(valid_extensions) and os.path.isfile(file):
            logger.info(f"Found existing image '{file}' in root directory. Using it.")
            return file
            
    # Fallback to generating mock image
    return create_mock_sports_image("mock_match_pitch.jpg")


def main() -> None:
    """Main execution block."""
    logger.info("Initializing Peter Drury Vision-to-Commentary Pipeline...")
    
    # 1. Image preparation
    image_path = find_or_create_image()

    # 2. Config & Instantiation
    vision_cfg = VisionConfig()
    commentary_cfg = CommentaryConfig(
        base_model_id="mlx-community/Phi-3-mini-4k-instruct-4bit",
        adapter_path="./drury_adapters",
        temperature=0.85,
        top_p=0.9,
        max_tokens=200
    )

    analyzer = VisionAnalyzer(vision_cfg)
    engine = DruryCommentaryEngine(commentary_cfg)
    pipeline = CommentaryPipeline(analyzer, engine)

    # 3. Execution
    try:
        print("\n" + "="*80)
        print("PIPELINE STARTING: Run local VLM -> Pipe to MLX SLM -> Poetic Commentary Stream")
        print("="*80 + "\n")
        
        commentary_stream = pipeline.run(image_path)

        print("PETER DRURY MONOLOGUE STREAM:")
        print("-" * 35)
        for token in commentary_stream:
            print(token, end="", flush=True)
        print("\n" + "-" * 35 + "\n")

        logger.info("Pipeline executed successfully.")

    except Exception as e:
        logger.critical(f"Pipeline crashed with an unhandled exception: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
