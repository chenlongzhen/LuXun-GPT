# -*- coding: utf-8 -*-
# @Time : 2023/7/29 14:28
# @Author : chenlongzhen
# @File : fin_instruction.py
# @function :

import pandas as pd
import json
import random
from tqdm import tqdm


def generate_random_instruction():
	base_instruction = "你是民生银行信用卡客服，针对客户的问题进行回答"
	instruction_variations = [
		"作为民生银行信用卡客服，您的任务是回答客户的问题",
		"您是民生银行信用卡客服，负责解答客户的疑问",
		"在这个场景中，您是民生银行信用卡客服，需要对客户的问题进行回复",
		"作为民生银行信用卡客服人员，您需要回答客户的问题",
		"您扮演着民生银行信用卡客服的角色，要回应客户提出的问题",
	]
	return random.choice(instruction_variations)


def csv_to_jsonl(input_file, output_file):
	df = pd.read_csv(input_file, encoding='utf-8')
	df_best = df[(df['is_best'] == 1) & (df['reply'].notnull())]
	
	json_list = []
	for _, row in tqdm(df_best.iterrows()):
		instruction = generate_random_instruction()
		context = f"Instruction: {instruction}\nInput: {row['title']}\nAnswer: "
		target = row['reply']
		json_item = {"context": context, "target": target}
		json_list.append(json_item)
	
	with open(output_file, 'w', encoding='utf-8') as f:
		for item in json_list:
			json.dump(item, f,  ensure_ascii=False)
			f.write('\n')


if __name__ == "__main__":
	input_file = "example_data/financezhidao_filter.csv"
	output_file = "example_data/fin_output.jsonl"
	csv_to_jsonl(input_file, output_file)
	
