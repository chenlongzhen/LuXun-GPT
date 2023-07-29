
# cd到当前项目主目录
cd $dirname $0;
cd ..
pwd

# preprocess
python ./fin_instruction.py \
    --data_path ./example_data/financezhidao_filter.csv \
    --save_path ./example_data/fin_output.jsonl

# tokenize
python ./tokenize_dataset.py \
    --jsonl_path ./example_data/fin_output.jsonl \
    --save_path ./example_data/fin_dataset \
    --max_seq_length 400 