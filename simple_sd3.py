"""
This script provides an ImageGenerator class to generate and navigate through images using Stable Diffusion.
It includes a Gradio interface for interacting with the image generation and navigation functionalities.

Classes:
    ImageGenerator

Functions:
    main()

Copyright (c) 2024 Hideyuki Inada
"""

import os
import sys
import time
import json
import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline
from PIL.PngImagePlugin import PngInfo
from image_navigator import ImageNavigator
from image_utils import read_image
from file_utils import get_image_file_list_in_directory
from typing import List, Optional, Tuple

SKIP_NUM = 10
OUTPUTS_DIR = "outputs"
CHECKPOINT_DIR = "checkpoint"  # local cache for "stabilityai/stable-diffusion-3-medium-diffusers"

class ImageGenerator:
    """
    A class to generate and navigate through images using Stable Diffusion.

    Methods:
        rescan_output_directory() -> None
        generate_button_handler(prompt: str, negative_prompt: str, steps: int, guidance_scale: float) -> Tuple
        setup_ui(file_paths: Optional[List[str]] = None) -> gr.Blocks
        __call__() -> None
    """

    def __init__(self, checkpoint_dir: Optional[str] = None, outputs_dir: Optional[str] = None):
        """
        Initializes the ImageGenerator with the specified checkpoint and output directories.

        Args:
            checkpoint_dir (Optional[str]): The directory for model checkpoints.
            outputs_dir (Optional[str]): The directory to save generated images.
        """
        self.checkpoint_dir = checkpoint_dir
        self.outputs_dir = outputs_dir
        self.file_paths = None
        self.current_index = 0
        self.pipe = None

        if outputs_dir is None:
            raise ValueError("Output directory is not specified.")
        os.makedirs(outputs_dir, exist_ok=True)

        self.pipe = StableDiffusion3Pipeline.from_pretrained(CHECKPOINT_DIR, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

        # Full path, file name, creation time
        self.file_paths = get_image_file_list_in_directory(self.outputs_dir)
        self.nav = ImageNavigator(
            app=self,
            file_paths=self.file_paths,
            skip_num=SKIP_NUM,
            callback=read_image
        )
        self.gradio_app = self.setup_ui()

    def rescan_output_directory(self) -> None:
        """
        Rescans the output directory. This is to be called by the navigation class when an image is deleted.
        """
        self.file_paths = get_image_file_list_in_directory(self.outputs_dir)

    def generate_button_handler(self, prompt: str, negative_prompt: str, steps: int = 28, guidance_scale: float = 7.0) -> Tuple:
        """
        Generates an image based on the provided prompts, steps, and guidance scale.

        Args:
            prompt (str): The positive prompt for image generation.
            negative_prompt (str): The negative prompt for image generation.
            steps (int): The number of inference steps. Default is 28.
            guidance_scale (float): The guidance scale for image generation. Default is 7.0.

        Returns:
            Tuple: A tuple containing the generated image, file information, and metadata.
        """
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        file_name = str(time.time())
        output_path = os.path.join(self.outputs_dir, file_name + ".png")

        generation_parameters = {
            "time": time.time(),
            "positive_prompt": prompt,
            "negative_prompt": negative_prompt,
            "sampling_iterations": steps,
            "cfg": guidance_scale
        }
        str_generation_params = json.dumps(generation_parameters)
        metadata = PngInfo()
        metadata.add_text("generation_data", str_generation_params)
        image.save(output_path, pnginfo=metadata)

        # Update paths
        self.file_paths.append(output_path)
        self.current_index = len(self.file_paths) - 1  # Set to last
        self.nav.update_file_paths(self.file_paths, new_current_index=self.current_index)
        return read_image(output_path)  # This is to return consistent meta data

    def setup_ui(self, file_paths: Optional[List[str]] = None) -> gr.Blocks:
        """
        Sets up the Gradio user interface for the image generator.

        Args:
            file_paths (Optional[List[str]]): The list of image file paths.

        Returns:
            gr.Blocks: The Gradio Blocks object representing the UI.
        """
        css = "#output_image {height:800px}"
        with gr.Blocks(analytics_enabled=False, css=css) as app:
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        prompt = gr.TextArea(label="Positive prompt:", elem_id="small-textarea", lines=4, max_lines=4)
                    with gr.Row():
                        negative_prompt = gr.TextArea(label="Negative prompt:", elem_id="small-textarea", lines=4, max_lines=4)
                    with gr.Row():
                        guidance_scale = gr.Slider(value=7.0, minimum=0.0, maximum=30.0, step=0.1, label="Guidance scale")
                    with gr.Row():
                        steps = gr.Slider(value=28, minimum=1, maximum=150, step=1, label="Steps")
                    with gr.Row():
                        generate_btn = gr.Button("Generate")
                with gr.Column():
                    with gr.Row():
                        global image_field
                        image_field = gr.Image(label="Output Image", elem_id="output_image")
                    with gr.Row():
                        skip_backward_btn = gr.Button("<<")
                        prev_btn = gr.Button("<")
                        next_btn = gr.Button(">")
                        skip_forward_btn = gr.Button(">>")
                        first_btn = gr.Button("First")
                        last_btn = gr.Button("Last")
                    with gr.Row():
                        delete_btn = gr.Button("Delete")
                    with gr.Row():
                        gr.Markdown("File info:")
                        file_info = gr.Markdown()
                    with gr.Row():
                        gr.Markdown("File metadata:")
                        metadata = gr.HTML()

            # Define the list of fields where output of a function is set.
            # This needs to match the return value for the target function.
            output = [
                image_field, file_info, metadata
            ]

            # Event handlers
            generate_btn.click(fn=self.generate_button_handler,
                            inputs=[prompt, negative_prompt, steps, guidance_scale],
                            outputs=output,
                            api_name="generate")

            # Navigation
            skip_backward_btn.click(fn=self.nav.skip_backward_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="skip_backward")
            prev_btn.click(fn=self.nav.prev_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="prev")
            next_btn.click(fn=self.nav.next_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="next")
            skip_forward_btn.click(fn=self.nav.skip_forward_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="skip_forward")
            first_btn.click(fn=self.nav.first_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="first")
            last_btn.click(fn=self.nav.last_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="last")
            delete_btn.click(fn=self.nav.delete_button_handler,
                            inputs=None,
                            outputs=output,
                            api_name="delete")
            
        return app

    def __call__(self) -> None:
        """
        Launches the Gradio application.
        """
        self.gradio_app.launch()

def main() -> None:
    """
    The main function to create an instance of ImageGenerator and launch the application.
    """
    image_generator = ImageGenerator(
        checkpoint_dir=CHECKPOINT_DIR,
        outputs_dir=OUTPUTS_DIR
    )
    image_generator()


if __name__ == "__main__":
    main()
