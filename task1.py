import os
import pandas as pd
import ast
import re

tags = pd.read_csv("data/tags.csv")
sentence = open("data/sentences.txt", 'r')
ouput_file = "results/task_1_output.tsv"

tags_dict = {row[1]:ast.literal_eval(row[2]) for row in tags.values}
flip_tags = {}
multi_kw = {}
sen_tag = []

untagged_sen = 0

# preprocessing of the text
def clean(text):
	text = re.sub(r'[^\w\s\$\/]', '', str.lower(str.strip(text)))
	return text

def write_tsv(ouput_file, sen_tag):
	with open(ouput_file, "w") as f:
	    for sen, tags in sen_tag:
	        if tags:
	            f.write(f"{sen}\t{', '.join(sorted(tags))}\n")
	        else:
	            f.write(f"{sen}\t\n")

# creation of tags dict
for tag, kws in tags_dict.items():
	for kw in kws:
		kw = clean(kw)
		if ' ' in kw:
			first_word = kw.split()[0]
			multi_kw.setdefault(first_word, []).append((kw, tag))
		flip_tags.setdefault(kw, []).append(tag)

# main loop to assign tags
for sen in sentence.readlines():
    sen = str.strip(sen)
    tags = []
    clean_sen = clean(sen).split()
    idx = 0
    len_ = len(clean_sen)
    while idx < len_:

        if clean_sen[idx] in flip_tags:
            tags.extend(flip_tags[clean_sen[idx]])

        elif clean_sen[idx] in multi_kw:
            for phrase, tag in multi_kw[clean_sen[idx]]:
                len_kw = len(phrase.split())
                if ' '.join(clean_sen[idx: idx+len_kw]) == phrase:
                    tags.append(tag)
                    idx += len_kw - 1  
			

        idx += 1
    
    tags = list(set(tags))
    sen_tag.append((sen, tags))
    if not tags:
        untagged_sen += 1
		
# Stats
print(f"Total tagged sentences: {len(sen_tag) - untagged_sen}")
print(f"Total un-tagged sentences: {untagged_sen}", end="\n\n")

# creating and saving tsv
write_tsv(ouput_file, sen_tag)
print(f"Tagged and saved successfully to {ouput_file}")
