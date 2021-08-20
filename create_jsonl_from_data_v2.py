import json
import os
import codecs
import html
import csv
import pandas as pd
import re
import numpy as np
import math
import spacy
'''
use spacy to create jsonl-like file
'''
nlp = spacy.load("en_core_web_sm")
# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
# for token in doc:
#     print(type(token), token.idx)
#
# exit()

"""
conll和txt合併成類似squad的格式
"""

squad = {"data": [], "version": "v2.0"}
input_file_dir = "../original_data/"

train_file_0 = "BioCreative_TrainTask3.0.tsv"
train_file_1 = "BioCreative_TrainTask3.1.tsv"
SMM4H18_file = "SMM4H18_Train.csv"

val_file = "BioCreative_ValTask3.tsv"

# train_files = [train_file_0, train_file_1]
train_files = [SMM4H18_file]
val_files = [val_file]
out_fname = "train_data_SMM_v1.jsonl"   ##########
question = "extract the spans that mention a medication or dietary supplement in tweets."
qid_count = 0

question_tokens = [[t.text, t.idx] for t in nlp(question)]

temp_json_list = [{"header": {"dataset": "SQuAD", "split": "train"}}]   #############

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
        if file in train_files:  ##############################
            in_file_path = os.path.join(r, file)
            print(in_file_path)
            df = pd.read_csv(in_file_path, delimiter='\t')
            # df = df.replace(np.nan, '', regex=True)
            for index, row in df.iterrows():

                processed_str = process_text(row['text'])
                # print(processed_str)
                # start_idx = None
                # answer_text = None
                answer_text_list = []  # may have multiple answer in SMM4H18
                start_idx_list = []
                sent_tokens = nlp(processed_str)
                context_tokens = [[t.text, t.idx] for t in sent_tokens]
                # for token in sent_tokens:
                #     print(token.text, token.i, token.idx)

                # temp_json = {"id": "", "context": processed_str,
                #              "context_tokens": [[t.text, t.idx] for t in sent_tokens]}
                # print(temp_json)

                # print(processed_str)
                if 'start' in row:  # original data
                    answer_text = row['span']
                    # if no answer in dataset, replace it with random str so we won't find it in text
                    if answer_text == '-':
                        answer_text = "rand0m 5tr1ng"
                    # print(type(answer_text), repr(answer_text))
                    start_idx = processed_str.find(answer_text)

                    answer_text_list.append(answer_text)
                    start_idx_list.append(start_idx)

                else:  # SMM external data
                    answer_text = row['Drug Name Extracted']
                    # if no answer in dataset, replace it with random str so we won't find it in text
                    if not isinstance(answer_text, str):
                        answer_text = "rand0m 5tr1ng"
                    # print(type(answer_text), repr(answer_text))
                    for ans in answer_text.split(";"):
                        ans = ans.strip()
                        start_idx = processed_str.lower().find(ans.lower())
                        real_ans = processed_str[start_idx:start_idx + len(ans)]  # because SMM dataset sometimes lowercase,
                        # print(repr(ans), repr(real_ans))
                        if real_ans == '' and ans != 'rand0m 5tr1ng':
                            print(repr(ans), index)
                        answer_text_list.append(real_ans)
                        start_idx_list.append(start_idx)
                # print(answer_text, start_idx)

                for idx_, answer_text_ in enumerate(answer_text_list):
                    answer_text = answer_text_
                    start_idx = start_idx_list[idx_]

                    title = str(row['user_id'])  ###############################
                    qas = []
                    if start_idx != -1:
                        detected_answers = [
                            {"text": answer_text,
                             "char_spans": [],
                             "token_spans": []
                             }]  #

                        answer_tokens = nlp(answer_text)
                        answer_start_idx = -1

                        # print(type(answer_tokens))
                        answer_token_len = len(answer_tokens)  # token數量 通常為單字數量

                        for token_idx in range(len(sent_tokens)):  # 找到token位於原本spacy.doc的哪幾個
                            span = sent_tokens[token_idx:token_idx + answer_token_len]  # 根據spacy斷句的answer token決定span長度
                            # print("checking span:", span.text, token_idx)

                            if span.text == answer_tokens.text:  # 有在doc的index，需額外找到token字元的index
                                # print(span.text, token_idx)

                                answer_start_idx = span.doc[span.start].idx
                                # print(answer_start_idx)  # token於doc index位置
                                detected_answers[0]["char_spans"].append([answer_start_idx, answer_start_idx + len(answer_text) - 1])
                                detected_answers[0]["token_spans"].append([span.start, span.start + answer_token_len - 1])
                                break

                        # print(detected_answers)
                        qas.append(
                            {
                                "answers": [answer_text],
                                "question": question,
                                "id": title + "_" + str(qid_count),
                                "qid": str(qid_count),
                                "question_tokens": question_tokens,
                                "detected_answers": detected_answers
                            }
                        )
                    else:  # no answer
                        qas.append(
                            {
                                "answers": [""],  # not inputting "rand0m 5tr1ng"
                                "question": question,
                                "id": title + "_" + str(qid_count),
                                "qid": str(qid_count),
                                "question_tokens": question_tokens,
                                "detected_answers": {}
                            }
                        )
                    qid_count += 1

                    paragraphs = {
                                   "id": "",
                                   "context": processed_str,
                                   "qas": qas,
                                   "context_tokens": context_tokens
                                   }  # only one paragraph
                    temp_json_list.append(paragraphs)

with open(out_fname, "w", encoding='utf8') as f:
    for json_ in temp_json_list:
        json.dump(json_, fp=f, ensure_ascii=False)
        f.write("\n")
