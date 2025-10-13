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
        self.get_logger().info(f"Loading BLIP2 model on {self.device}...")
        model: Blip2ForImageTextRetrievalModified = (
            Blip2ForImageTextRetrievalModified.from_pretrained(
                model_path, dtype=torch.float16
            )
            .to(self.device)
            .eval()
        )
        self.get_logger().info(f"Model dtype: {model.dtype}")
        self.get_logger().info(f"Model device: {model.device}")
        self.model = model
        self.get_logger().info("Loading processor...")
        processor: AutoProcessor = AutoProcessor.from_pretrained(
            model_path, use_fast=True
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
    logging.basicConfig(level=logging.INFO)

    model_path = "/home/ghz/blip2-itm-vit-g"
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    extractor = Blip2ImageFeatureExtractor(model_path=model_path)
    features = extractor.process_image(
        image, "Open the door with the vertical black handle"
    )
    print(features.shape)  # [1, 768]
