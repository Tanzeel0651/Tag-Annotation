import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer, util


def encode(model, sentence_or_list):
    if isinstance(sentence_or_list, list):
        return model.encode(sentence_or_list, normalize_embeddings=True)
    else:
        return model.encode([sentence_or_list], normalize_embeddings=True)[0]

def write_tsv(ouput_file, sen_tag):
	with open(ouput_file, "w") as f:
	    for sen, tags in sen_tag:
	        if tags:
	            f.write(f"{sen}\t{', '.join(sorted(tags))}\n")
	        else:
	            f.write(f"{sen}\t\n")
                
def read_tsv(file):
    sen_tag = []
    with open(file, 'r') as file:
        for line in file.readlines():
            sen, tag = line.split("\t")
            if not tag in ["\n"]:
                sen_tag.append((sen, [x.strip() for x in tag.split(',')]))
            else:
                sen_tag.append((sen, []))
    return sen_tag

def add_tags(x,y):
    if not x and not y:
        return None
    elif not x:
        return y
    elif not y:
        return x
    else:
        return list(set(x+y))

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read sentences
with open("data/sentences.txt", "r") as f:
    sentences = [line.strip() for line in f.readlines()]

sentence_embeddings = encode(model, sentences)
print(f"Encoded {len(sentences)} sentences.")

# Read tags and keywords
tags_df = pd.read_csv("data/tags.csv")
tags_dict = {row["name"]: ast.literal_eval(row["keywords"]) for _, row in tags_df.iterrows()}

# Encode tags (keywords + tag name itself)
tag_embeddings = {}
for tag, keywords in tags_dict.items():
    full_context = keywords + [tag]
    tag_embeddings[tag] = encode(model, full_context)
    tag_embeddings[tag] = np.mean(tag_embeddings[tag], axis=0)

print(f"Encoded {len(tag_embeddings)} tags.")


# Compute Similarities
results = []
for i, sentence_embed in enumerate(sentence_embeddings):
    similarities = {}
    for tag, tag_embed in tag_embeddings.items():
        sim = util.cos_sim(sentence_embed, tag_embed).item()
        similarities[tag] = sim

    sorted_tags = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    top_tags = [tag for tag, score in sorted_tags if score > 0.45]  # threshold 
    results.append((sentences[i], top_tags))
    
 
    
# keywords Mapped
mixed_mapping = [] 
kw_map = read_tsv("results/task_1_output.tsv")
for x,y in zip(results, kw_map):
    if x[0] == y[0]:
        mixed_mapping.append((x[0], add_tags(x[1], y[1])))
    else:
        print("Cosine mismatch:", x)
        print("Keywords mismatch:",y)


# Save Results

import pdb;pdb.set_trace()
write_tsv("results/task_2_cosine.tsv", mixed_mapping)

print("Saved cosine similarity results to results/task_2_cosine.tsv")
