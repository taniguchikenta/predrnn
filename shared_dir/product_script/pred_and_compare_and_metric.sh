export CUDA_VISIBLE_DEVICES="0"
cd ..
python -u run.py \
	--is_training 0 \
    --device cuda \
    --dataset_name products \
    --train_data_paths ./data/products_test \
    --valid_data_paths ./data/products_test \
    --test_data_paths ./data/products_test \
    --test_dir_name true \
    --save_dir checkpoints/test \
    --gen_frm_dir results/test/true \
    --model_name predrnn_v2 \
    --visual 0 \
    --reverse_input 1 \
    --img_width 256 \
    --img_channel 3 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --num_save_samples 100 \
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
    --pretrained_model ./checkpoints/train/products_test/model.ckpt-70000
    
cd compare_imgs
python3 MY_lpips_metric.py \
    --metric_img_dir dataset/data1 \
    --save_directory result/data1_MY \
    --anomaly_value 0.4 \
