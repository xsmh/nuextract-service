import json
from transformers import AutoModelForCausalLM, AutoTokenizer


def predict_NuExtract(model, tokenizer, text, schema, example=["","",""]):
    schema = json.dumps(json.loads(schema), indent=4)
    input_llm =  "<|input|>\n### Template:\n" +  schema + "\n"
    for i in example:
      if i != "":
          input_llm += "### Example:\n"+ json.dumps(json.loads(i), indent=4)+"\n"
    
    input_llm +=  "### Text:\n"+text +"\n<|output|>\n"
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=4000).to("cpu")

    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    return output.split("<|output|>")[1].split("<|end-output|>")[0]


model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-tiny", trust_remote_code=True)

model.to("cpu")

model.eval()

text = """"""

schema = """{
    "Traction": {
        "Github Stars": "",
        "Number of Users": "Look for total installs, sign-ups, or users",
        "Active Users / Engagement": "Look for community platform members, daily/weekly/monthly active users and mention what platform",
        "Revenue": "revenue, subscriptions, or paying customers",
    }
    }"""

prediction = predict_NuExtract(model, tokenizer, text, schema, example=["","",""])
print(prediction)
