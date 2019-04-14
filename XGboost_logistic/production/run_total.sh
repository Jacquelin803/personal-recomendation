#!/usr/bin/env bash
python='/Users/jacquelin/anaconda3/bin/python'
original_train_data='../data/train.txt'
original_test_data='../data/test.txt'
train_data='../data/train_file'
test_data='../data/test_file'
tree_model='../data/xgb.model'
lr_coef_mix_model='../data/lr_coef_mix_model'
tree_mix_model='../data/xgb_mix_model'
feature_num_file='../data/feature_num'
model_type='lr_gbdt'    #lr_gbdt or gbdt,生成单独树模型要改为gbdt

if [ -f $original_train_data -a -f $original_test_data ]; then
    $python ana_train_data.py $origin_train_data $origin_test_data $train_data $test_data $feature_num_file
else
    echo 'no origin file'
    exit
fi

if [ -f $train_data ]; then
    if [ $model_type == 'gbdt' ]; then
        $python train.py $train_data $feature_num_file $tree_model
    fi
    if [ $model_type == 'lr_gbdt' ]; then
        $python train.py $train_data $feature_num_file $tree_mix_model $lr_coef_mix_model
    fi
else
    echo 'no train file'
    exit
fi

if [ $model_type == 'gbdt' ]; then
    if [ -f $tree_model ]; then
        $python check.py $test_data $tree_model $feature_num_file
    else
        echo 'no gbdt model file'
        exit
    fi
elif [ $model_type == 'lr_gbdt' ]; then
    if [ -f $tree_mix_model -a -f $lr_coef_mix_model ]; then
        $python check.py $test_data $tree_mix_model $lr_coef_mix_model $feature_num_file
    else
        echo 'no lr_gbdt model file'
        exit
    fi
else
    echo 'wrong model type'
fi


