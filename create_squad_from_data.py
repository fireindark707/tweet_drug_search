import json
import os
import codecs
import html
import csv
import pandas as pd
import re
import numpy as np
import math
"""
tsv, csv合併成類似squad的格式
"""

squad = {"data": [], "version": "v2.0"}
input_file_dir = "../original_data/"

train_file_0 = "BioCreative_TrainTask3.0.tsv"
train_file_1 = "BioCreative_TrainTask3.1.tsv"
SMM4H18_file = "SMM4H18_Train.csv"

val_file = "BioCreative_ValTask3.tsv"

train_files = [train_file_0, train_file_1, SMM4H18_file]
val_files = [val_file]
out_fname = "train.json"
question = "extract the spans that mention a medication or dietary supplement in tweets."
qid_count = 0


def process_text(s):
    username_re = "@[a-zA-Z0-9_]{0,15}"
    url_re = "https?://(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9]{1,6}/([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    url_re1 = "http(s)?://t\.co/\w+"

    # replace &amp; &lt; ...
    new_s = html.unescape(s)
    # replace username @abc123
    new_s_2 = re.sub(username_re, "[name]", new_s)
    # replace url
    new_s_3 = re.sub(url_re, "[url]", new_s_2)

    return new_s_3

for r, d, f in os.walk(input_file_dir):
    for file in f:
        # if file.endswith(".tsv") or file.endswith(".csv"):
        if file in train_files:   ##############################
            in_file_path = os.path.join(r, file)
            print(in_file_path)
            df = pd.read_csv(in_file_path, delimiter='\t')
            # df = df.replace(np.nan, '', regex=True)
            for index, row in df.iterrows():
                processed_str = process_text(row['text'])
                # print(processed_str)
                start_idx = None
                answer_text = None

                # print(processed_str)
                if 'start' in row:  # original data
                    answer_text = row['span']
                    # if no answer in dataset, replace it with random str so we won't find it in text
                    if answer_text == '-':
                        answer_text = "rand0m 5tr1ng"
                    # print(type(answer_text), repr(answer_text))
                    start_idx = processed_str.find(answer_text)

                else:  # HMM external data
                    answer_text = row['Drug Name Extracted']
                    # if no answer in dataset, replace it with random str so we won't find it in text
                    if not isinstance(answer_text, str):
                        answer_text = "rand0m 5tr1ng"
                    # print(type(answer_text), repr(answer_text))
                    start_idx = processed_str.find(answer_text)

                # print(answer_text, start_idx)

                title = str(row['user_id'])  ###############################
                qas = []
                if start_idx != -1:
                    qas.append(
                        {"question": question,
                         "id": title + "_" + str(qid_count),
                         "answers": [{"text": answer_text, "answer_start": start_idx}],
                         "is_impossible": False
                         }
                    )
                else:  # no answer
                    qas.append(
                        {"question": question,
                         "id": title + "_" + str(qid_count),
                         "answers": [],
                         "is_impossible": True
                         }
                    )
                qid_count += 1

                paragraphs = [{"qas": qas,
                               "context": processed_str
                               }]  # only one paragraph
                squad["data"].append({"title": title, "paragraphs": paragraphs})


with open(out_fname, "w", encoding='utf8') as f:
    json.dump(squad, fp=f, indent=4, ensure_ascii=False)
