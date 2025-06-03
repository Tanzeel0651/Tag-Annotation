import re
import pandas as pd
import ast
import numpy as np

# import gensim.downloader as api
# from gensim.models import Word2Vec
# from gensim.models import KeyedVectors


import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset


def read_tsv(file):
    sen_tag = []
    with open(file, 'r') as file:
        for line in file.readlines():
            sen, tag = line.split("\t")
            if not tag in ["\n"]:
                sen_tag.append((sen, [x.strip() for x in tag.split(',')]))
    return sen_tag
 

def write_tsv(ouput_file, sen_tag_list):
	with open(ouput_file, "w") as f:
	    for sen, tags in sen_tag_list:
	        if tags:
	            f.write(f"{sen}\t{', '.join(sorted(tags))}\n")
	        else:
	            f.write(f"{sen}\t\n")
    
def clean(sentences):
    all_sen = []
    for sentence in sentences:
        sentence = re.sub(r'[^\w\s\$\/]', ' ', str.lower(str.strip(sentence)))
        all_sen.append(sentence)
    return all_sen


def get_mlb(tag_file="data/tags.csv", none_class="NO_TAG"):
    tag_classes = list(pd.read_csv(tag_file)["name"]) + [none_class]
    mlb = MultiLabelBinarizer(classes=tag_classes)
    return mlb

class ChatDataset(Dataset):
    def __init__(self, tokenizer_name, max_length, mlb, complete=False):
        self.complete = complete
        self.mlb = mlb
        
        if complete:
            with open("data/sentences.txt", "r") as f:
                self.sentences = [line.strip() for line in f.readlines()]
            tags = ["NO_TAG"]*len(self.sentences)
        
        else:
            sen_tag = read_tsv("results/task_1_output.tsv")
            tags = [tags for _, tags in sen_tag]
            self.sentences = [sen for sen, _ in sen_tag]
            
     
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
        self.tokenized_corpus = self.tokenizer(clean(self.sentences), truncation=True, 
                                               padding="max_length", max_length=max_length,
                                               return_tensors="pt")
        self.y = self.mlb.fit_transform(tags)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return {
                "sentence": self.sentences[idx],
                "input_ids": self.tokenized_corpus["input_ids"][idx],
                "attention_mask": self.tokenized_corpus["attention_mask"][idx],
                "labels": torch.tensor(self.y[idx], dtype=torch.float32),
            }
        
class BERTclassifer(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.pooler_output)
        logits = self.fc(x)
        return logits
    
    
def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    sentences = []
    
    for item in batch:
        sentences.append(item["sentence"])
        input_ids.append(item["input_ids"])
        attention_mask.append(item["attention_mask"])
        labels.append(item["labels"])
        
    return {
            "sentence": sentences,
            "input_ids": torch.stack(input_ids).to("cuda"),
            "attention_mask": torch.stack(attention_mask).to("cuda"),
            "labels": torch.stack(labels).to("cuda")
        }


def train(model, dataloader, optimizer, num_epochs, device="cuda"):
    model = model.to(device)
    total_batches = len(dataloader)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if step%2==0:
                print(f"\rEpoch: {epoch+1}/{num_epochs} -- Batch: ({step+1}/{total_batches}) -- Loss: {loss.item():.4f} -- Epoch Loss: {epoch_loss:.4f}", end="", flush=True)
            epoch_loss += loss.item()
            
        print()
        


def evaluate(model, dataloader, mlb, device="cuda"):
    model = model.to(device)
    model.eval()
    predictions = []
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            sentence = batch["sentence"]
            
            outputs = model(input_ids = input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs)
            binary_preds = (probs > 0.35).cpu().numpy()
            
            tags = mlb.inverse_transform(binary_preds)
            # import pdb;pdb.set_trace()
            for sent, tag_list in zip(sentence, tags):
                predictions.append((sent, tag_list))
                
            if step%2==0:
                print(f"\rBatch: ({step+1}/{total_batches})", end="", flush=True)
                
        print()
    return predictions


def main():
    # Hyperparameters
    model_name = "bert-base-uncased"
    batch_size = 8
    max_epochs = 5
    max_length = 150
    learning_rate = 2e-5
    
    # dataset & dataloaders
    mlb = get_mlb()
    train_dataset = ChatDataset(model_name, max_length, mlb, complete=False)
    complete_dataset = ChatDataset(model_name, max_length, mlb, complete=True)
    
    print("Dataset Loaded")
    print(f"Train Dataset size: {len(train_dataset)}")
    

    train_dataloader = torch.utils.data.DataLoader(
                            dataset = train_dataset,
                            batch_size = batch_size,
                            shuffle = True,
                            collate_fn = lambda batch:collate_fn(batch)
    )
    
    eval_dataloader = torch.utils.data.DataLoader(
                            dataset = complete_dataset,
                            batch_size = batch_size,
                            shuffle = False,
                            collate_fn = lambda batch: collate_fn(batch)
    )
    num_classes = complete_dataset[0]["labels"].shape[0]
    
    # Model + Optimizer
    model = BERTclassifer(model_name, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # train
    print("Starting Training")
    train(model=model, 
          dataloader=train_dataloader, 
          num_epochs=max_epochs,
          optimizer=optimizer)
    
    # evaluate
    print("Starting Evaluation")
    predictions = evaluate(model, eval_dataloader, mlb)
    
    return predictions


if __name__=="__main__":
    output_file = "results/task_2_output.tsv"
    predictions = main()
    write_tsv(output_file, predictions)
    print(f"Tagged and saved successfully to {output_file}")
            





