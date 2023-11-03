export CUDA_VISIBLE_DEVICES="0"
set -ex
cd ..
python3 MY_lpips_metric.py \
    --metric_img_dir dataset/data1 \
    --save_directory result/data1_MY \
    --anomaly_value 0.4 \
    
    
#    --real_img dataset/data_mine_3/real \
#    --fake_img dataset/data_mine_3/fake \
#    --save_directory result/3_bin256_5box_32strip_anom75 \
#    --bin_num 256 \
#    --anomaly_value 0.75 \
#    --hsplit 32 \
#    --vsplit 5 \
    
    
    
    
    
#    --real_img dataset/data_mine_1/real \
#    --fake_img dataset/data_mine_1/fake \
#    --save_directory result/result_mine_1_bin64_box_all_test_fourtest \
#    --hsplit 16 \
#    --vsplit 4 \
