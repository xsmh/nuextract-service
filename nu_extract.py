import json
from transformers import AutoModelForCausalLM, AutoTokenizer


def predict_nuextract(model, tokenizer, text, nu_schema, example=["","",""]):
    nu_schema = json.dumps(nu_schema, indent=4)
    input_llm =  "<|input|>\n### template:\n" +  nu_schema + "\n"
    for i in example:
      if i != "":
          input_llm += "### example:\n"+ json.dumps(json.loads(i), indent=4)+"\n"
    
    input_llm +=  "### text:\n"+text +"\n<|output|>\n"
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to("cpu")
    
    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    output_str = output.split("<|output|>")[1].split("<|end-output|>")[0]
    output_dict = json.loads(output_str)
    return output_dict


def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained("numind/nuextract-tiny", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("numind/nuextract-tiny", trust_remote_code=True)
    model.to("cpu")
    model.eval()
    return model, tokenizer
