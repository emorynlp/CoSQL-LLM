from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
import torch
import argparse
from transformers.trainer_utils import set_seed # type: ignore

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_path1', type = str, default = '')
    parser.add_argument('--llm_path2', type = str, default = '')
    parser.add_argument('--save_path', type = str, default = '')


    opt = parser.parse_args()

    return opt
if __name__ == "__main__":
    set_seed(42)
    opt = parse_option()
    print(opt)
    save_path = opt.save_path

    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path1) # type: ignore
    model1 = AutoModelForCausalLM.from_pretrained(opt.llm_path1, device_map = "auto", torch_dtype = torch.float16) # type: ignore
    model2 = AutoModelForCausalLM.from_pretrained(opt.llm_path2, device_map = "auto", torch_dtype = torch.float16) # type: ignore
    model1 = model1.to("cpu") # type: ignore
    model2 = model2.to("cpu") # type: ignore


    print(model1.state_dict().keys() == model2.state_dict().keys()) # type: ignore
    
    # Average the weights
    averaged_state_dict = {}
    for key in model1.state_dict(): # type: ignore
        # print(key)
        averaged_state_dict[key] = (model1.state_dict()[key] + model2.state_dict()[key]) / 2 # type: ignore
    print('loading model again')
    merged_model = AutoModelForCausalLM.from_pretrained(opt.llm_path1, device_map = "auto", torch_dtype = torch.float16) # type: ignore
    
    print('loading new weights into model')
    merged_model.load_state_dict(averaged_state_dict) # type: ignore

    print('saving')
    tokenizer.save_pretrained(save_path) # type: ignore

    merged_model.save_pretrained( # type: ignore
        save_path,
        max_shard_size = "100GB"
    )
    print(f"Merged model saved to {save_path}")

    tokenizer = AutoTokenizer.from_pretrained(save_path) # type: ignore
    print('loading merged model')
    model = AutoModelForCausalLM.from_pretrained(save_path, device_map = "auto", torch_dtype = torch.float16) # type: ignore
    print('done')