from typing import Optional, Tuple
from transformers import Blip2ForImageTextRetrieval
from transformers import AutoProcessor
import torch
import torch.nn as nn
import numpy as np
import logging


class Blip2ForImageTextRetrievalModified(Blip2ForImageTextRetrieval):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        use_image_text_matching_head: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds: torch.Tensor = vision_outputs[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        if use_image_text_matching_head:
            raise NotImplementedError
        else:
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=return_dict,
            )
            image_embeds = (
                query_outputs[0] if not return_dict else query_outputs.last_hidden_state
            )
            image_embeds = image_embeds.to(dtype=self.vision_projection.weight.dtype)

            if self.config.image_token_index is not None:
                input_ids = input_ids[:, self.config.num_query_tokens :]
                attention_mask = attention_mask[:, self.config.num_query_tokens :]

            query_embeds = self.embeddings(
                input_ids=input_ids,
            )
            text_outputs = self.qformer(
                query_embeds=query_embeds,
                query_length=0,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            question_embeds: torch.Tensor = (
                text_outputs[0] if not return_dict else text_outputs.last_hidden_state
            )
            question_embeds = question_embeds.to(
                dtype=self.text_projection.weight.dtype
            )

            # normalized features
            image_embeds_proj = nn.functional.normalize(
                self.vision_projection(image_embeds), dim=-1
            )
            text_embeds_proj = nn.functional.normalize(
                self.text_projection(question_embeds[:, 0, :]), dim=-1
            )

            # cosine similarity as logits
            logits_per_image = torch.matmul(image_embeds_proj, text_embeds_proj.t())

            # print("Debug info:")
            # print(f"{logits_per_image.shape=}")
            # print(f"{image_embeds.shape=}")
            # print(f"{question_embeds.shape=}")
            # print(f"{image_embeds_proj.shape=}")
            # print(f"{text_embeds_proj.shape=}")

            if not return_dict:
                return (
                    logits_per_image,
                    image_embeds,
                    question_embeds,
                    image_embeds_proj,
                    text_embeds_proj,
                    # vision_outputs,
                    # query_outputs,
                    # text_outputs,
                )


class Blip2ImageFeatureExtractor:
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.dtype = torch.float16
        self.model_path = model_path

    def load_model(self):
        self.get_logger().info(f"Loading BLIP2 model on {self.device}...")
        model: Blip2ForImageTextRetrievalModified = (
            Blip2ForImageTextRetrievalModified.from_pretrained(
                self.model_path, dtype=self.dtype
            )
            .to(self.device)
            .eval()
        )
        self.get_logger().info(f"Model dtype: {model.dtype}")
        self.get_logger().info(f"Model device: {model.device}")
        self.model = model
        self.get_logger().info("Loading processor...")
        processor: AutoProcessor = AutoProcessor.from_pretrained(
            self.model_path, use_fast=True
        )
        self.processor = processor
        self.get_logger().info("Initialized.")

    def process_image(
        self, image: np.ndarray, prompt: str, project: bool = True
    ) -> torch.Tensor:
        with torch.inference_mode():
            inputs: dict = self.processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(self.device, self.model.dtype)
            outputs: Tuple[torch.Tensor, ...] = self.model(**inputs, return_dict=False)
            (
                logits_per_image,
                image_embeds,
                question_embeds,
                image_embeds_proj,
                text_embeds_proj,
            ) = outputs
            similarity = logits_per_image.squeeze(-1)

            # [1, num_queries]
            attention_weights = nn.functional.softmax(similarity, dim=-1)
            # [1, num_queries, 1]
            attention_weights_expanded = attention_weights.unsqueeze(-1)
            # [1, 768] or [1, 256]
            image_features = image_embeds_proj if project else image_embeds
            weighted_image_embeds = (image_features * attention_weights_expanded).sum(
                dim=1
            )
            # print("Final weighted image embeds shape:", weighted_image_embeds.shape)
            return weighted_image_embeds

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__name__)


if __name__ == "__main__":
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
    )

    # TODO: maybe there can be a reversed dataset that writes to MCAP?
    from mcap_data_loader.serialization.flb import (
        McapFlatBuffersWriter,
        FlatBuffersSchemas,
    )
    from time import time_ns
    from pathlib import Path
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-directory", "-id", type=str, help="Directory containing the dataset"
    )
    parser.add_argument(
        "--output-directory",
        "-od",
        type=str,
        default="",
        help="Directory to save the output",
    )
    parser.add_argument("--model-path", "-mp", type=str, help="Path to the BLIP2 model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_path = args.model_path
    extractor = Blip2ImageFeatureExtractor(model_path=model_path)
    extractor.load_model()

    input_dir = Path(args.input_directory)
    keys = [
        "/env_camera/color/image_raw",
    ]
    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(
            data_root=input_dir,
            keys=keys,
            strict=False,
        )
    )
    dataset.load()
    output_dir = args.output_directory
    if output_dir == "":
        output_dir = input_dir.parent / f"{input_dir.name}_blip2_features"
    else:
        output_dir = Path(output_dir)
    for index, episode in enumerate(dataset):
        print(
            f"Processing episode {index + 1}/{len(dataset)}: {episode.config.data_root}"
        )
        writer = McapFlatBuffersWriter()
        writer.create_writer(output_dir / episode.config.data_root.name, overwrite=True)
        for key in keys:
            writer.register_channel(f"{key}/features", FlatBuffersSchemas.FLOAT_ARRAY)
        for sample in tqdm(episode, desc="Processing samples", total=len(episode)):
            # pprint(sample)
            for key, value in sample.items():
                features = extractor.process_image(
                    value, "Open the door with the vertical black handle"
                ).squeeze(0)
                # print(features.shape)
                writer.add_array(
                    f"{key}/features", features.tolist(), time_ns(), time_ns()
                )
        writer.unset_writer(finish=True)
    print("Done")
