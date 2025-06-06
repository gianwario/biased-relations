python 3_modify_activation.py \
    --bert_model bert-base-uncased \
    --data_path ../data/PARAREL/data_all.json \
    --tmp_data_path ../data/biased_relations/biased_relations_all_bags.json \
    --kn_dir ../results/bert-uncased/kn \
    --output_dir ../results/bert-uncased/
    --gpus 1 \
    --max_seq_length 128 \
    --debug 100000 \