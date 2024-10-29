import json
import logging
from dataclasses import dataclass, field
import os
import hydra
from tqdm import tqdm
from vllm import LLM, SamplingParams

LOG = logging.getLogger(__name__)


@dataclass
class CheckContaminationConfig:
    """Top-level parameters for the script"""

    input_file: str  # Output of the retrieve_similar.py script
    output_file: str  # Where to save the generations
    # Prompt configuration
    prompt_template: str | None = None  # Path to the prompt template file
    prompt_config: str = "judge/check-contamination"
    examples_type: str | None = None  # To customize few-shot examples
    # Inference parameters
    model_name_or_path: str = "gpt2"  # Model name or path for vLLM
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 128
    generation_key: str = "contaminated"
    retrieve_key: str = "problem"  # Used to fill in prompt with retrieve_key1 and retrieve_key2
    skip_filled: bool = False  # Skip already filled generations
    check_both_ways: bool = False  # Check contamination in both directions
    dry_run: bool = False  # If True, only print the first prompt without running generation

    def __post_init__(self):
        if self.prompt_template is None:
            # Default prompt template if none is provided
            self.prompt_template = (
                """  Help me determine if the following two planning tasks in given are paraphrased.

First planning task: {problem1}
Second planning task: {problem2}

Disregard the names and minor changes in word order that appear within.

If the two tasks are very similar (paraphrased) and if they produce the same answer in corresponding conditions, we consider them to be the same task.

Give a brief explanation and respond with "True" (tasks are the same) or "False" (tasks are different). Do not respond with anything else."""
            )


def setup_logging():
    logging.basicConfig(level=logging.INFO)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="check_contamination_config", node=CheckContaminationConfig)


@hydra.main(version_base=None, config_name="check_contamination_config")
def main(cfg: CheckContaminationConfig):
    check_contamination(cfg)


def check_contamination(cfg: CheckContaminationConfig):
    LOG.info("Config used: %s", cfg)

    # Initialize the vLLM engine with the specified model
    llm_engine = LLM(model=cfg.model_name_or_path,
                    download_dir='/workspace-SR003.nfs2/huggingface/hub/',
                    tensor_parallel_size=8,
                    gpu_memory_utilization=0.92,
                    )

    # Prepare the prompt template
    prompt_template = (
                """Help me determine if the following two planning tasks in given are the same.

First planning task: {problem1}
Second planning task: {problem2}

Disregard the names and minor changes in condition that are almost not affecting the final plan appear within.

If the two tasks are very similar (paraphrased) and if they produce the same answer in corresponding conditions, we consider them to be the same task.

Respond with "True" (tasks are the same) or "False" (tasks are different). Dont generate anything else"""
    )

    LOG.info("Prompt used: %s", prompt_template)

    # Load data
    with open(cfg.input_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    first_element = {
        f"{cfg.retrieve_key}1": data[0][cfg.retrieve_key],
        f"{cfg.retrieve_key}2": data[0]["similar_items"][0],
    }

    # Create the prompt by filling in the template
    example_prompt = prompt_template.format(**first_element)
    LOG.info("Example prompt:\n%s", example_prompt)

    if cfg.dry_run:
        return

    data_points = []

    # Adjust batch size
    top_k = max([len(i["similar_items"]) for i in data])
    print(top_k)
    cfg.batch_size = max(1, cfg.batch_size // top_k // (2 if cfg.check_both_ways else 1))

    starting_idx = 0
    if cfg.skip_filled:
        try:
            with open(cfg.output_file, "rt", encoding="utf-8") as fin:
                starting_idx = len(fin.readlines())
        except FileNotFoundError:
            LOG.warning(f"File `{cfg.output_file}` not found, starting from scratch")
    data = data[starting_idx:]
    total = 0
    num_contaminated = 0
    
    output_dir = os.path.dirname(cfg.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    

    with open(cfg.output_file, "at" if cfg.skip_filled else "wt", encoding="utf-8", buffering=1) as fout:
        for idx, data_point in enumerate(
            tqdm(data, initial=starting_idx, total=len(data) + starting_idx)
        ):
            data_point.pop(cfg.generation_key, None)
            data_points.append(data_point)

            if len(data_points) == cfg.batch_size or idx == len(data) - 1:
                # Construct data for LLM calls
                all_data = []
                for original_data_point in data_points:
                    for similar_item in original_data_point["similar_items"]:
                        all_data.append(
                            {
                                f"{cfg.retrieve_key}1": original_data_point[cfg.retrieve_key],
                                f"{cfg.retrieve_key}2": similar_item,
                            }
                        )

                        if cfg.check_both_ways:
                            all_data.append(
                                {
                                    f"{cfg.retrieve_key}2": original_data_point[cfg.retrieve_key],
                                    f"{cfg.retrieve_key}1": similar_item,
                                }
                            )

                prompts = [prompt_template.format(**dp) for dp in all_data]

                # Prepare sampling parameters
                sampling_params = SamplingParams(
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens,
                )

                
                
                tokenizer = llm_engine.get_tokenizer()
                prompts = [tokenizer.apply_chat_template([{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True) for message in prompts]
                # Generate outputs using vLLM
                outputs = llm_engine.generate(prompts, sampling_params)

                # Process the outputs
                output_idx = 0
                for original_data_point in data_points:
                    all_generations = []
                    elem = {}
                    contaminated = False
                    num_generations = len(original_data_point["similar_items"]) * (2 if cfg.check_both_ways else 1)
                    for _ in range(num_generations):
                        output = outputs[output_idx]
                        generation = output.outputs[0].text
                        all_generations.append(generation)
                        if "True" in generation.strip():
                            contaminated = True
                        output_idx += 1
                    elem[cfg.generation_key] = contaminated
                    if contaminated:
                        num_contaminated += 1
                    total += 1
                    elem["all_generations"] = all_generations
                    elem.update(original_data_point)
                    fout.write(json.dumps(elem) + "\n")
                data_points = []
    if total > 0:
        LOG.info(
            "Contamination portion: %.2f%% (%d/%d)",
            100 * num_contaminated / total,
            num_contaminated,
            total,
        )


if __name__ == "__main__":
    setup_logging()
    main()

