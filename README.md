# Tag Annotation

## Overview

This project focuses on semi-automated sentence classification for a domain-specific dataset. The goal is to assign one or more relevant category tags to sentences using both rule-based and ML-based methods. The full corpus consists of **2,997 sentences**.

We experimented with two primary tagging strategies:

---

## Task 1: Keyword-Based Matching

In this stage, we applied direct keyword and phrase matching techniques based on a curated dictionary of tags and associated keywords.

* **Total Sentences Tagged**: 1,545 out of 2,997
* **Method**: Cleaned text matched against keyword/tag mappings using exact and multi-word phrase matching.

### Tag Distribution (Matching)

```
{'Personal Loans': 43,
 'Death of a Relative': 8,
 'Commercial Banking': 26,
 'Vehicle Loan': 67,
 'Fraud': 113,
 'Retirement Planning': 15,
 'Cash Management': 26,
 'Marital Status': 6,
 'Commercial Loans': 7,
 'Complaints': 4,
 'Trust/Estate Planning': 8,
 'Aging Parents': 2,
 'Credit Card': 74,
 'Starting a Business': 1,
 'Follow up requested': 0,
 'Payments': 59,
 'Debit Cards': 47,
 'Children and Savings': 6,
 'Statements': 60,
 'Security Messages': 10,
 'Retail Banking': 486,
 'Online Accounts': 436,
 'Home Loans': 53,
 'Investing': 42,
 'College/Independence': 2,
 'Tax Information': 45,
 'Accounts': 210}
```

---

## Task 2: BERT Classification & Cosine Similarity

We tested BERT for multi-label classification, but due to extreme label imbalance and limited training data (1,460 labeled examples), the model performance was not optimal.

To address this, we combined:

1. **Sentence embeddings using SentenceTransformer**
2. **Cosine similarity with averaged tag + keyword embeddings**

This hybrid approach significantly improved coverage:

* **Total Sentences Tagged**: 2,081 out of 2,997

### Tag Distribution (Hybrid Cosine + Keyword)

```
{'Personal Loans': 148,
 'Death of a Relative': 13,
 'Commercial Banking': 53,
 'Vehicle Loan': 75,
 'Fraud': 153,
 'Retirement Planning': 24,
 'Cash Management': 136,
 'Marital Status': 10,
 'Commercial Loans': 52,
 'Complaints': 33,
 'Trust/Estate Planning': 25,
 'Aging Parents': 13,
 'Credit Card': 229,
 'Starting a Business': 15,
 'Follow up requested': 0,
 'Payments': 174,
 'Debit Cards': 123,
 'Children and Savings': 20,
 'Statements': 63,
 'Security Messages': 18,
 'Retail Banking': 671,
 'Online Accounts': 665,
 'Home Loans': 65,
 'Investing': 70,
 'College/Independence': 8,
 'Tax Information': 55,
 'Accounts': 646}
```

---

## Key Challenges

* **Imbalanced Label Distribution**: Several tags had 0 or fewer than 10 examples, making supervised learning difficult.
* **Overfitting Risk with BERT**: Due to small dataset size.
* **Lack of negative class examples ("no tag")**: Difficult for classifier to learn when not to assign a tag.

---

## Future Improvements

To improve tagging quality, especially for underrepresented categories:

1. **Synthetic Data Augmentation**

   * Generate new sentences for low-sample tags using LLMs (e.g., ChatGPT)

2. **Zero-shot Tagging + Manual Labeling**

   * Use LLMs in zero-shot mode to pre-label and curate examples.

3. **RAG with Vector DB**

   * Store all sentences in a vector DB (e.g., FAISS, Chroma).
   * For each tag, retrieve nearest neighbors using semantic search and review manually.

---

## Final Thoughts

Keyword-based techniques provide precision, while embedding-based cosine similarity adds generalization. A hybrid of both allows robust, scalable, and interpretable sentence classification in real-world, label-sparse settings.

**Directory Structure**

```
├── data/
│   ├── sentences.txt
│   └── tags.csv
├── results/
│   ├── task_1_output.tsv
│   └── task_2_cosine.tsv
├── task1_keyword_match.py
├── task2_cosine_similarity.py
├── task2_bert.py
└── README.md
```
