import json
import random
import logging
from pathlib import Path
from typing import List
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


few_shot_examples = [
    {
        "problem": "Task: Replace the glass in the bedroom door.\nConditions: Bedroom. The door to the bedroom is closed. The door is equipped with a handle. The door is equipped with old glass. The light in the bedroom is off. The light switch is located to the right of the door. By the wall in the bedroom, there is a new glass for the door.",
        "augmented_problem": "Task: Install a frosted glass panel in the office door.\nConditions: Office. The office door is ajar and has an outdated glass panel with scratches. The office light is dim. A ladder and the new frosted glass panel are positioned near the door. A tool kit containing a screwdriver and glass cutters is on the desk."
    },
    {
        "problem": "Task: Paint the walls in the living room.\nConditions: Living room. The walls are currently white. There is a can of blue paint and paintbrushes on the floor. The room is empty of furniture. Windows are open for ventilation.",
        "augmented_problem": "Task: Apply wallpaper to the walls in the study room.\nConditions: Study room. The walls are painted beige and have minor marks. A roll of patterned wallpaper, glue, and a brush are placed on the study table. The room is ventilated with the windows fully open, and furniture is pushed to the center of the room."
    },
    {
        "problem": "Task: Install the new faucet in the bathroom sink.\nConditions: Bathroom. The old faucet is in place. There is a wrench and a screwdriver on the sink counter. The water supply to the sink is turned off. A new faucet is still in its box on the counter.",
        "augmented_problem": "Task: Replace the showerhead with a rain shower model in the guest bathroom.\nConditions: Guest bathroom. The old showerhead is mounted and leaking. A replacement rain showerhead is on a shelf along with an adjustable wrench and sealing tape. The main water supply to the bathroom is off, and the space is well-lit with an overhead light."
    },
    {
        "problem": "Task: Assemble the new desk in the office.\nConditions: Office. The desk parts are on the floor. There are screws and a screwdriver nearby. The assembly instructions are on the table.",
        "augmented_problem": "Task: Set up a new wardrobe in the bedroom.\nConditions: Bedroom. The wardrobe pieces, including panels and doors, are laid out on the carpet. A set of bolts, screws, and an Allen key are in a small box on the dresser. Assembly instructions are printed and propped up on a chair, and the floor area is cleared."
    },
    {
        "problem": "Task: Mount the new TV on the living room wall.\nConditions: Living room. The TV and mounting bracket are on the floor. A power drill and screws are nearby. The wall is free of decorations.",
        "augmented_problem": "Task: Secure a heavy-duty mirror on the hallway wall.\nConditions: Hallway. The mirror and wall anchors are leaning against the wall. A power drill and stud finder are on a nearby side table. The wall has a single coat hook, which is positioned on the opposite side of where the mirror will be mounted."
    }
]

def load_dataset(dataset_path: str) -> List[dict]:
    dataset = []
    with open(dataset_path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def main():
    # Paths and configurations
    dataset_path = '/home/jovyan/medvedev/NeMo-Skills/nemo_skills/contamination-results.jsonl'  # Replace with your dataset path
    output_path = './augmented_problems2.jsonl'     # Output file for augmented problems
    model_path = 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF'            # Replace with your model path

    # Load your dataset
    logger.info('Loading dataset...')
    dataset = load_dataset(dataset_path)

    # Initialize the vLLM model
    logger.info('Initializing the model...')
    llm = LLM(model=model_path,
                    download_dir='/workspace-SR003.nfs2/huggingface/hub/',
                    tensor_parallel_size=8,
                    gpu_memory_utilization=0.92,
                    )
    tokenizer = llm.get_tokenizer()

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=512,
        n=64  # Number of problem variants to generate
    )

    augmented_problems = []

    logger.info('Starting problem augmentation...')
    for idx, data in tqdm(enumerate(dataset)):
        logger.info(f'Augmenting problem {idx + 1}/{len(dataset)}')

        # Prepare the prompt with few-shot examples
        prompt = "Write a new planning task and condition inspired by a given one. Make the new problem reasonable and solvable in your produced conditions.\nYou are given several examples: \n"
        for example in few_shot_examples:
            prompt += f"Original Problem:\n{example['problem']}\n\nAugmented Problem:\n{example['augmented_problem']}\n\n"

        # Add the current problem
        current_problem = data['problem']  
        prompt += f"Original Problem:\n{current_problem}\n\n"
        
        prompt += """
Write another problem inspired by this one.
Don't just change the numbers and context, but try to create a problem that requires another approach to solve.
Start directly with the problem statement and DO NOT include any phrases such as "Here is a new math problem inspired by a given one".
After the problem is completed finish your response right away.
Make sure not to solve the problem after formpulating it
"""

        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        # Generate new problems
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        generated_outputs = outputs[0].outputs  # List of GenerationOutput

        # Collect generated problems
        for gen_output in generated_outputs:
            new_problem = gen_output.text.strip()
            augmented_problems.append({
                'original_problem': current_problem,
                'augmented_problem': new_problem
            })

    # Save the augmented problems
    logger.info(f'Saving augmented problems to {output_path}...')
    with open(output_path, 'w') as f:
        for item in augmented_problems:
            json.dump(item, f)
            f.write('\n')

    logger.info('Problem augmentation completed.')

if __name__ == '__main__':
    main()
