cd ..

#CUDA_VISIBLE_DEVICES=0 \
python ./lora_finetune.py \
    --dataset_path ./example_data/fin_dataset \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 30000 \
    --save_steps 5000 \
    --learning_rate 1e-5 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir ./saved_models_07291911


