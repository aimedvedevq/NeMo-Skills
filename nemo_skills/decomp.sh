python3 /home/jovyan/medvedev/NeMo-Skills/nemo_skills/inference/check_contamination.py \
    input_file=/home/jovyan/medvedev/NeMo-Skills/nemo_skills/math-contamination-retrieved.jsonl \
    output_file=./contamination-results.jsonl \
    model_name_or_path=nvidia/Llama-3.1-Nemotron-70B-Instruct-HF \
    batch_size=128 \
    max_tokens=256 \
    temperature=0.7 \
    top_p=0.95 \
    skip_filled=False \
    check_both_ways=False \
    dry_run=False