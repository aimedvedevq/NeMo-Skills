# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# partially adapted from https://github.com/lm-sys/llm-decontaminator/tree/main

import json
import logging
import os
import sys
from typing import Any

import hydra
import torch
from sentence_transformers import SentenceTransformer, util

from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging, unroll_files

LOG = logging.getLogger(__file__)


def top_k_similarity(from_emb, to_emb, top_k, similarity_threshold=0.90):
    cosine_scores = util.cos_sim(to_emb, from_emb)
    
    # Get top K results first
    top_k_results = torch.topk(cosine_scores, k=top_k, dim=1)
    
    # Store filtered indices and scores
    top_k_filtered_indices = []
    top_k_filtered_scores = []

    for i in range(cosine_scores.size(0)):
        indices = []
        scores = []

        # Retrieve top K results first
        for j in range(top_k_results.indices.size(1)):
            indices.append(top_k_results.indices[i][j].item())
            scores.append(top_k_results.values[i][j].item())
        
        # Check if any additional items meet or exceed the threshold
        above_threshold_items = (cosine_scores[i] >= similarity_threshold).nonzero(as_tuple=True)[0].tolist()
        
        # If there are more than 5 items above the threshold, take all those items
        if len(above_threshold_items) >= 5:
            indices = above_threshold_items
            scores = [cosine_scores[i][idx].item() for idx in above_threshold_items]
        
        # Collect final indices and scores (either top 5 or all above threshold)
        top_k_filtered_indices.append(indices[:max(len(indices), 5)])  # Ensures at least 5 are included
        top_k_filtered_scores.append(scores[:max(len(scores), 5)])
    
    return top_k_filtered_indices, top_k_filtered_scores


def encode(model, data, batch_size):
    clean_data = [str(item['problem']) for item in data if 'problem' in item]  # Extracts 'problem' field
    return model.encode(clean_data, batch_size=batch_size, show_progress_bar=True)


def read_data(file_paths, retrieve_key) -> list:
    all_data = []
    for file_path in unroll_files(file_paths):
        with open(file_path, 'rt', encoding='utf-8') as file:
            all_data.extend([json.loads(line) for line in file])
    return all_data


@nested_dataclass
class RetrieveSimilarConfig:
    # will find top_k most similar examples in retrieve_from files for each example in compare_to files
    retrieve_from: Any
    compare_to: Any

    # where to save the final file with most similar examples
    # will have the same number of rows as the number of unique "retrieve_key" instances in compare_to files
    output_file: str

    # the model used to compute embedding, default is sentence transformer
    model: str = 'multi-qa-MiniLM-L6-cos-v1'

    # how many most-similar examples to retrieve
    top_k: int = 3
    retrieve_key: str = 'problem'

    batch_size: int = 512

    def __post_init__(self):
        if isinstance(self.retrieve_from, str):
            self.retrieve_from = self.retrieve_from.split(" ")

        if isinstance(self.compare_to, str):
            self.compare_to = self.compare_to.split(" ")


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_retrieve_similar_conifg", node=RetrieveSimilarConfig)


@hydra.main(version_base=None, config_name="base_retrieve_similar_conifg")
def retrieve_similar(cfg: RetrieveSimilarConfig):
    cfg = RetrieveSimilarConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    os.makedirs(os.path.dirname(cfg.output_file), exist_ok=True)

    model = SentenceTransformer(cfg.model)

    retrieve_from_list = read_data(cfg.retrieve_from, cfg.retrieve_key)
    compare_to_list = read_data(cfg.compare_to, cfg.retrieve_key)

    retrieve_from_embeddings = encode(model, retrieve_from_list, batch_size=cfg.batch_size)
    compare_to_embeddings = encode(model, compare_to_list, batch_size=cfg.batch_size)
    top_k_indices, top_k_scores = top_k_similarity(retrieve_from_embeddings, compare_to_embeddings, cfg.top_k, similarity_threshold=0.90)
    top_k_similar_items = []
    for i, compare_item in enumerate(compare_to_list):
        similar_items = [retrieve_from_list[index] for index in top_k_indices[i]]
        similarity_scores = top_k_scores[i]
        top_k_similar_items.append(
            {
                **compare_item,
                'similar_items': [item["problem"] for item in similar_items],
                'similarity_scores': similarity_scores,
            }
    )

    with open(cfg.output_file, 'w', encoding='utf-8') as fout:
        for entry in top_k_similar_items:
            fout.write(json.dumps(entry) + '\n')


HELP_MESSAGE = get_help_message(RetrieveSimilarConfig)


if __name__ == "__main__":
    if '--help' in sys.argv or '-h' in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        retrieve_similar()
