from typing import Optional, Tuple, Dict
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

        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f"Total parameters: {total_params:,}")
        # print(f"Trainable parameters: {trainable_params:,}")

        self.get_logger().info("Loading processor...")
        processor: AutoProcessor = AutoProcessor.from_pretrained(
            self.model_path, use_fast=True
        )
        self.processor = processor
        self.get_logger().info("Initialized.")

    def process_image(self, image: np.ndarray, prompt: str) -> Dict[str, torch.Tensor]:
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
            wie_list = []
            for image_features in (image_embeds, image_embeds_proj):
                weighted_image_embeds = (
                    image_features * attention_weights_expanded
                ).sum(dim=1)
                wie_list.append(weighted_image_embeds)
            # print("Final weighted image embeds shape:", weighted_image_embeds.shape)
            return {"features": wie_list[0], "features_proj": wie_list[1]}

    def get_logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__name__)


if __name__ == "__main__":
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
        McapFlatBuffersSampleDataset,
    )

    # TODO: maybe there can be a reversed dataset that writes to MCAP?
    from mcap_data_loader.serialization.flb import (
        McapFlatBuffersWriter,
        FlatBuffersSchemas,
    )
    from mcap_data_loader.utils.av_coder import DecodeConfig
    from time import time_ns, monotonic
    from pathlib import Path
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor as PoolExecutor, as_completed
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
    parser.add_argument(
        "--prompt",
        "-prom",
        type=str,
        help="Prompt for feature extraction",
        # default="Open the cabinet door",
        default="Open the drawer with the black handle",
        # default="Open the drawer with the black handle",
    )
    parser.add_argument(
        "--keys",
        "-k",
        type=str,
        nargs="+",
        help="Keys to extract features from",
        default=[
            "/env_camera/color/image_raw",
            "/follow_camera/color/image_raw",
        ],
    )
    parser.add_argument("--model-path", "-mp", type=str, help="Path to the BLIP2 model")
    parser.add_argument(
        "--num-workers", "-nw", type=int, default=4, help="Number of workers"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_path = args.model_path
    extractor = Blip2ImageFeatureExtractor(model_path=model_path)
    extractor.load_model()

    input_dir = Path(args.input_directory)
    keys = args.keys
    dataset = McapFlatBuffersEpisodeDataset(
        McapFlatBuffersEpisodeDatasetConfig(
            data_root=input_dir,
            keys=keys,
            strict=False,
            media_configs=[DecodeConfig(mismatch_tolerance=5, frame_format="rgb24")],
        )
    )
    dataset.load()
    output_dir = args.output_directory
    if output_dir == "":
        output_dir = input_dir.parent / f"{input_dir.name}_blip2_features"
    else:
        output_dir = Path(output_dir)
    executor = PoolExecutor(max_workers=args.num_workers)

    def process_episode(index, episode: McapFlatBuffersSampleDataset):
        print(
            f"Processing episode {index + 1}/{len(dataset)}: {episode.config.data_root}"
        )
        writer = McapFlatBuffersWriter()
        writer.create_writer(output_dir / episode.config.data_root.name, overwrite=True)
        for key in keys:
            writer.register_channel(f"{key}/features", FlatBuffersSchemas.FLOAT_ARRAY)
            writer.register_channel(
                f"{key}/features_proj", FlatBuffersSchemas.FLOAT_ARRAY
            )
        for sample in tqdm(episode, desc="Processing samples", total=len(episode)):
            # pprint(sample)
            for key, value in sample.items():
                features_dict = extractor.process_image(
                    value["data"][:, :, ::-1].copy(), args.prompt
                )
                for feature_key, features in features_dict.items():
                    # print(features.shape)
                    writer.add_float_array(
                        f"{key}/{feature_key}",
                        features.squeeze(0).tolist(),
                        time_ns(),
                        time_ns(),
                    )
        writer.unset_writer(finish=True)

    start = monotonic()
    futures = []
    for index, episode in enumerate(dataset):
        futures.append(executor.submit(process_episode, index, episode))
    for index, future in enumerate(as_completed(futures)):
        future.result()
    executor.shutdown(wait=True)
    cost_time = monotonic() - start
    print(
        f"All done in {cost_time:.2f} seconds ({cost_time / (index + 1):.2f} seconds per episode)."
    )
