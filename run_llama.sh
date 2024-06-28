torchrun --nproc_per_node 1 main_interact_llama.py \
    --data_root /dccstor/bebo/codes/ScienceQA/data \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --caption_file data/caption_qcm_llava.json \
    --bs 32 --eval_bs 32 --epoch 20 --lr 8e-5 --output_len 64 \
    --use_generate --prompt_format QCMG-A \
    --output_dir experiments0620 \
     --ckpt_dir /dccstor/diac/cea/public_models/llama3/Meta-Llama-3-8B-Instruct/ \
    --tokenizer_path /dccstor/diac/cea/public_models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model \
    --max_seq_len 2048 --max_gen_len 16 --max_batch_size 32 --use_caption  --max_inter_gen_len 256 \
    --evaluate_dir models/mm-cot-base-answer \
    --test_le ./predictions_ans_test.json \
    # --use_caption

# --use_caption
# --caption_file data/instruct_captions.json \
# python
# instruct_captions