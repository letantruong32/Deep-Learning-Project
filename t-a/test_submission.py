import json, sys
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
import torch
import config


path = Path('')

def load_model():
    model = torch.load(path/'model')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval();
    return model

def encode(sequence):
    return tokenizer.encode_plus(
                sequence,
                add_special_tokens=True,
                max_length=512,
                return_token_type_ids=False,
                padding=True,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
    )
    
def eval(text):
    # This is where you call your model to get the number of stars output
    encoded = encode(text)
    with torch.no_grad():
        output = model(encoded['input_ids'].cpu(), token_type_ids=None, attention_mask=encoded['attention_mask'].cpu())[0]
        pred_flat = np.argmax(output, axis=1).flatten()
        sig_factor = torch.sigmoid(output) / torch.sigmoid(output).sum()
        return pred_flat.item() + 1
    
tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
model = load_model()

if len(sys.argv) > 1:
    validation_file = sys.argv[1]
    with open("output.jsonl", "w") as fw:
        with open(validation_file, "r") as fr:
            for line in fr:
                review = json.loads(line)
                fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
    print("Output prediction file written")
else:
    print("No validation file given")