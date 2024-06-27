export TIKTOKEN_CACHE_DIR="./tmp"

torchrun --nproc_per_node 1 main_interact_llama.py \
    --data_root /dccstor/niccoloav/mm-cot/ScienceQA/data\
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --caption_file /dccstor/niccoloav/mm-cot/data/instruct_captions.json \
    --bs 48 --eval_bs 48 --epoch 20 --lr 8e-5 --output_len 64 \
    --use_generate --prompt_format QCMG-A \
    --output_dir experiments0620 \
    --ckpt_dir /dccstor/diac/cea/public_models/llama3/Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path /dccstor/diac/cea/public_models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 2048 --max_gen_len 16 --max_batch_size 48 --use_caption --max_inter_gen_len 64 \
    --evaluate_dir models/mm-cot-base-answer \
    --test_le /dccstor/niccoloav/mm-cot/models/mm-cot-base-rationale/predictions_ans_test_original.json \
    --save_outputs /dccstor/niccoloav/mm-cot/llama_qcmg_a_llava_caption_qcm_greedy_.json \
    --temperature 0 \
    --top_p 0.01 \
    --seed 45
    # --use_caption

# --use_caption
# --caption_file data/instruct_captions.json \
# python
# instruct_captions