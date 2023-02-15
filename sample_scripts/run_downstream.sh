python main.py \
       --mode downstream \
       --model bert_downstream \
       --dataset custom \
       --name downstream_bert \
       --epochs 50 \
       --max_seq_len 20 \
       --batch_size 64 \
       --transformer_type bert-base-uncased \
       --learning_rate 3e-06 \
       --weight_decay 0.001 \
       --optim_type adam \
       --seed 123 \
       --early_stopping True \
       --patience 5 \
       --output_dir "<path/to/output_dir>" \
       --asset_dir "<path/to/dataset_dir>" \
       --custom_train_path "<path/to/rewritten/training/set>" \
       --custom_valid_path "<path/to/rewritten/training/set/if/available>" \
       --custom_test_path "<path/to/original/test/set>" \
       --downstream_test_data none
