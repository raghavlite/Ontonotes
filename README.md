# Ontonotes
Part Analysis

```
python parth_wise_analysis.py train.english.128_words.jsonlines
```

**Input** : train.english.128_words.jsonlines

**Output** : Analysis_FT_Ontonotes_{0.7,0.8,0.9}.json

**Download Input (and precomputed output)** from https://drive.google.com/drive/folders/1MTTeVf-4ICZDbJBAsvBo7DXIESozAz3m?usp=sharing

`Analysis_FT_Ontonotes_{0.7,0.8,0.9}.json` are the three analysis json files with docids as keys and different statistics as values. The 0.7, 0.8, 0.9 refers to the threshold of the string-matching threshold. We can find the merged clusters for a specific document by

```
stats = json.load(open(Analysis_FT_Ontonotes_0.7.json))
```

`stats[docid]["clusters"]` gives the merged propernoun clusters of each document across parts. Each cluster contains the propernoun mentions/spans. Some clusters are empty i.e. no propernoun mentions are present in the cluster. Each mention within this cluster is a list of spans. Each span again is a list of tokens.



