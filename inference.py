from transformers import AutoModel
import torch
import os

from transformers import AutoTokenizer

from peft import PeftModel
import argparse

def generate(instruction, text):
    with torch.no_grad():
        input_text = f"Instruction: {instruction}\nInput: {text}\nAnswer: "
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).cuda()
        output = peft_model.generate(
            input_ids=input_ids,
            max_length=256,
            do_sample=False,
            temperature=0.0,
            num_return_sequences=1
        )[0]
        output = tokenizer.decode(output)
        answer = output.split("Answer: ")[-1]
    return answer.strip()
    

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--base_model", type=str, default="model/glm")
    argparser.add_argument("--lora", type=str, default="/root/autodl-tmp/LuXun-GPT/saved_models")
    argparser.add_argument("--instruction", type=str, default="您扮演着民生银行信用卡客服的角色，要回应客户提出的问题")
    argparser.add_argument("--input_path", type=str, default="test_data/test.txt")
    argparser.add_argument("--output_path", type=str, default="test_data/test_output_v2.txt")
    argparser.add_argument("--interactive", action="store_true")

    args = argparser.parse_args()

    model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True, load_in_8bit=True, device_map='auto', revision="v0.1.0")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    if args.lora == "":
        print("#> No lora model specified, using base model.")
        peft_model = model.eval()
    else:
        print("#> Using lora model:", args.lora)
        peft_model = PeftModel.from_pretrained(model, args.lora).eval()
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.interactive:
        while True:
            text = input("Input: ")
            print(generate(args.instruction, text))
    else:
        with open(args.input_path, "r", encoding="utf-8") as f:
            input_texts = [line.strip() for line in f]
        output_texts = []
        for text in input_texts:
            output_texts.append(generate(args.instruction, text))
        with open(args.output_path, "a+", encoding="utf-8") as f:
            f.write("Model: " + args.lora + "\n")
            f.write("Instruction: " + args.instruction + "\n\n")
            for input_text, output_text in zip(input_texts, output_texts):
                f.write("Input: " + input_text + "\n")
                f.write("Output: " + output_text + "\n\n")
    
