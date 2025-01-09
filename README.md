# Position Debiasing Fine-Tuning for Causal Perception in Long-Term Dialogue (IJCAI 2024)

## System Requirements
```bash
conda create -n CPD python==3.10
conda activate CPD
pip install -r requirement.txt
```

## Data
You should download ESConv and msc datasets and place them in the datasets floder.

```aiignore

├── ESConv
│   ├── ESConv_test.jsonl
│   ├── ESConv_train.jsonl
│   └── ESConv_valid.jsonl
├── msc
    ├── msc_test.jsonl
    ├── msc_train.jsonl
    └── msc_valid.jsonl

```

## Get Causal data

- See the get_causal_data.py

## Training
- See the scripts/finetune_causal.sh

## Eval
- See the scripts/finetune_causal.sh and scripts/eval.sh

## Cite

```
@inproceedings{ijcai2024p692,
  title     = {Position Debiasing Fine-Tuning for Causal Perception in Long-Term Dialogue},
  author    = {Fan, Shixuan and Wei, Wei and Li, Wendi and Mao, Xian-Ling and Xie, Wenfeng and Chen, Dangyang},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {6261--6269},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/692},
  url       = {https://doi.org/10.24963/ijcai.2024/692},
}
```
