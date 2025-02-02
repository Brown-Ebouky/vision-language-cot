python llama_parser_ensemble.py \
    --data_root ScienceQA/data \
    --caption_file data/instruct_captions.json \
    --model declare-lab/flan-alpaca-base \
    --user_msg rationale --img_type vit \
    --bs 8 --eval_bs 1 --epoch 20 --lr 8e-5 --output_len 512 \
    --use_caption --use_generate --final_eval --prompt_format QCMUG-A \
    --output_dir experiments_qmcue_preds \
    --test_le /dccstor/niccoloav/mm-cot/experiment_qcmue_80_perc/rationale_declare-lab-flan-alpaca-base_vit_QCMU-UE_lr8e-05_bs8_op512_ep20/predictions_ans_test.json \
    --load_ua \
    --final_eval \
    --evaluate_dir /dccstor/niccoloav/mm-cot/experiments0620/rationale_declare-lab-flan-alpaca-base_vit_QCMUE-A_lr8e-05_bs8_op64_ep20