export CUDA_VISIBLE_DEVICES="1"
cd ..
python -u run.py \
	--is_training 1 \
    --device cuda \
    --dataset_name products \
    --train_data_paths ./data/products_test \
    --valid_data_paths ./data/products_test \
    --test_data_paths ./data/products_test \
    --save_dir checkpoints//train/products_test_3 \
    --gen_frm_dir results/valid/products_test_3 \
    --num_save_samples 10 \
    --model_name predrnn_v2 \
    --visual 0 \
    --reverse_input 1 \
    --img_width 256 \
    --img_channel 3 \
    --input_length 2 \
    --total_length 12 \
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
    --batch_size 1 \
    --max_iterations 80000 \
    --display_interval 100 \
	--test_interval 5000 \
    --snapshot_interval 5000 \
#    --pretrained_model ./checkpoints/products_predrnn_v2/model.ckpt-80000 \

    #--is_training は "0" or "1" の値をとり、１なら学習＆テスト、０ならテストのみであると考えられる
