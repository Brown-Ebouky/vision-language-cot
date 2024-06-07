# base
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --data_root ScienceQA/data \
#     --caption_file data/instruct_captions.json \
#     --model declare-lab/flan-alpaca-base \
#     --user_msg rationale --img_type vit \
#     --bs 8 --eval_bs 1 --epoch 20 --lr 8e-5 --output_len 512 \
#     --use_caption --use_generate --final_eval --prompt_format QCMU-E \
#     --output_dir experiments \
#     --evaluate_dir /dccstor/niccoloav/mm-cot/experiment_qcmue/rationale_declare-lab-flan-alpaca-base_vit_QCMU-UE_lr8e-05_bs8_op512_ep20


    # CUDA_VISIBLE_DEVICES=0 python main.py \
    # --data_root ScienceQA/data \
    # --caption_file data/instruct_captions.json \
    # --model declare-lab/flan-alpaca-base \
    # --user_msg rationale --img_type vit \
    # --bs 8 --eval_bs 1 --epoch 20 --lr 8e-5 --output_len 512 \
    # --use_caption --use_generate --final_eval --prompt_format QCMU-E \
    # --output_dir experiments_qmcue_preds \
    # --test_le models/mm-cot-base-rationale/predictions_ans_test.json \
    # --load_ua \
    # --method "preds" \
    # --final_eval \
    # --evaluate_dir /dccstor/niccoloav/mm-cot/experiment_qcmue/rationale_declare-lab-flan-alpaca-base_vit_QCMU-UE_lr8e-05_bs8_op512_ep20

# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --data_root ScienceQA/data \
#     --caption_file data/instruct_captions.json \
#     --model declare-lab/flan-alpaca-base \
#     --user_msg rationale --img_type vit \
#     --bs 8 --eval_bs 16 --epoch 20 --lr 8e-5 --output_len 64 \
#     --use_caption --use_generate --prompt_format QCMG-A \
#     --output_dir experiments \
#     --eval_le models/mm-cot-base-rationale/predictions_ans_eval.json \
#     --test_le models/mm-cot-base-rationale/predictions_ans_test.json \
#     --evaluate_dir models/mm-cot-base-ans

    CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_root ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --img_type vit \
    --bs 8 --eval_bs 1 --epoch 20 --lr 8e-5 --output_len 512 \
    --use_caption --use_generate --final_eval --prompt_format QCMUE-A \
    --output_dir experiments_qmcue_preds \
    --test_le /dccstor/niccoloav/mm-cot/experiment_qcmue/rationale_declare-lab-flan-alpaca-base_vit_QCMU-UE_lr8e-05_bs8_op512_ep20/predictions_ans_test.json \
    --load_ua \
    --final_eval \
    --evaluate_dir /dccstor/niccoloav/mm-cot/experiments0620/rationale_declare-lab-flan-alpaca-base_vit_QCMUE-A_lr8e-05_bs8_op64_ep20

# # large
# # rationale generation
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
#     --data_root data/ScienceQA/data \
#     --caption_file data/instruct_captions.json \
#     --model declare-lab/flan-alpaca-large \
#     --user_msg rationale --img_type vit \
#     --bs 2 --eval_bs 4 --epoch 50 --lr 5e-5 --output_len 512 \
#     --use_caption --use_generate --prompt_format QCM-E \
#     --output_dir experiments \
#     --evaluate_dir models/mm-cot-large-rationale

# # answer inference
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main_central.py \
#     --data_root data/ScienceQA/data \
#     --caption_file data/instruct_captions.json \
#     --model declare-lab/flan-alpaca-large \
#     --user_msg answer --img_type vit \
#     --bs 4 --eval_bs 8 --epoch 50 --lr 5e-5 --output_len 64 \
#     --use_caption --use_generate --prompt_format QCMG-A \
#     --output_dir experiments \
#     --eval_le models/mm-cot-large-rationale/predictions_ans_eval.json \
#     --test_le models/mm-cot-large-rationale/predictions_ans_test.json \
#     --evaluate_dir models/mm-cot-large-answer 