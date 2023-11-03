export CUDA_VISIBLE_DEVICES="0"
cd ..
python -u run.py \
	--is_training 1 \
    --device cuda \
    --dataset_name products \
    --train_data_paths ./data/products_test_2 \
    --valid_data_paths ./data/products_test_2 \
    --test_data_paths ./data/products_test_2 \
    --save_dir checkpoints//train/products_test_2 \
    --gen_frm_dir results/valid/products_test_2 \
    --num_save_samples 20 \
    --model_name predrnn_v2 \
    --visual 0 \
    --reverse_input 1 \
    --img_width 256 \
    --img_channel 3 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --decouple_beta 0.01 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 5000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2000 \
    --lr 0.0001 \
    --batch_size 3 \
    --max_iterations 80000 \
    --display_interval 100 \
	--test_interval 5000 \
    --snapshot_interval 5000 \
#    --pretrained_model ./checkpoints/products_predrnn_v2/model.ckpt-80000 \

    # --is_training は "0" or "1" の値をとり、１なら学習＆テスト、０ならテストのみであると考えられる