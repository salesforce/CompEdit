


NAME=
DATA=
RUN=1
python -u train.py \
     --train_file ${DATA}/train.csv \
     --validation_file ${DATA}/val.csv  \
     --do_train \
     --do_eval \
     --model_name_or_path facebook/bart-large \
     --metric_for_best_model eval_mean_rouge \
     --output_dir output/${NAME}_${RUN} \
     --per_device_train_batch_size 4 \
     --max_source_length 1024 \
     --generation_max_len 64 \
     --val_max_target_length 64 \
     --overwrite_output_dir \
     --per_device_eval_batch_size 4 \
     --gradient_accumulation_steps 2 \
     --predict_with_generate \
     --evaluation_strategy epoch \
     --num_train_epochs 10 \
     --save_strategy epoch \
     --logging_strategy epoch \
     --load_best_model_at_end \
     --compute_rouge_for_train True \
     --seed $RUN &> ${NAME}_${RUN}.out ;


python select_checkpoints.py output/