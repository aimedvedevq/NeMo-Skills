python3 inference/retrieve_similar.py \
    ++retrieve_from="/home/jovyan/medvedev/foundation_model/foundation_model/data/preprocessed/train/planner_train_small_onemessage.jsonl"  \
    ++compare_to="/home/jovyan/medvedev/foundation_model/foundation_model/data/preprocessed/test/planner_test_small_onemessage.jsonl" \
    ++output_file=./math-contamination-retrieved.jsonl \
    ++top_k=5