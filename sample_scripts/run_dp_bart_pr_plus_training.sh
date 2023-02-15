python main.py \
       --mode pretrain \
       --model dp_bart \
       --dataset openwebtext \
       --name dp_bart_pr_plus_training_epsilon250 \
       --pruning True \
       --gradually_increase_pruning False \
       --data_split_cutoff 2000000 \
       --iteration_cutoff 500 \
       --private True \
       --epsilon 250 \
       --delta 1e-06 \
       --max_seq_len 20 \
       --dp_module clip_value \
       --dp_mechanism gaussian \
       --epochs 50 \
       --batch_size 32 \
       --learning_rate 1e-05 \
       --clipping_constant 5 \
       --optim_type adam \
       --seed 42 \
       --transformer_type facebook/bart-base \
       --early_stopping False \
       --output_dir "<path/to/output_dir>" \
       --asset_dir "<path/to/dataset_dir>" \
       --pruning_index_path "<path/to/pruning_indices.pt>" \
       --last_checkpoint_path "<path/to/model.pt>"
