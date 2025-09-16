import json
import os
import re
import random

from nltk.tokenize import word_tokenize # type: ignore
from nltk import ngrams # type: ignore
from sql_metadata import Parser # type: ignore
from pyserini.search.lucene import LuceneSearcher # type: ignore
from utils.bridge_content_encoder import get_matched_entries
from utils.db_utils import get_db_schema
import argparse
import torch
import numpy as np
from utils.db_utils import get_db_schema_sequence, get_matched_content_sequence


from transformers import AutoModelForCausalLM, AutoTokenizer # type: ignore
from utils.load_sft_dataset import SFTSQLGenerationDataset
from utils.db_utils import check_sql_executability, detect_special_char
from torch.utils.data import DataLoader
from tqdm import tqdm
from sqlconstraint import SQLConstraint
from transformers.generation.beam_constraints import PhrasalConstraint
from schema_item_filter import SchemaItemClassifierInference, filter_schema


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_path', type = str, default = "./ckpts/codes-7b-cosql/ckpt-4892")
    parser.add_argument('--sic_path', type = str, default = "./sic_ckpts/sic_spider")
    parser.add_argument('--table_num', type = int, default = 6)
    parser.add_argument('--column_num', type = int, default = 10)

    # parser.add_argument('--dataset_path', type = str, default = "./data/sft_cosql_dev_text2sql.json")

    parser.add_argument('--max_tokens', type = int, default = 4096)
    parser.add_argument('--max_new_tokens', type = int, default = 256)
    parser.add_argument('--output_path', type = str, default = "pred_sqls.txt")
    
    opt = parser.parse_args()

    return opt

def post_process(sql, schema_items):
    # print(sql)
    # print(schema_items)
    # exit()
    sql = sql.replace("\n", " ")
    sql = sql.replace("> =", ">=")
    sql = sql.replace("< =", "<=")
    sql = sql.replace("! =", "!=")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`"+column_name+"`")

    while "``" in sql:
        sql = sql.replace("``", "`")

    return sql

def text2sql_func(model, inputs, tokenizer, max_new_tokens):
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_beams = 4,
            num_return_sequences = 4,
            # constraints = [SQLConstraint(db_path = "soccer_3", tokenizer=tokenizer)],
            # constraints = [PhrasalConstraint(tokenizer("select",add_special_tokens=False).input_ids)],
        )
    # print('input')
    # print(tokenizer.decode(inputs['input_ids'][0]))
    
    generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
    # print('output')
    # for sql in generated_sqls:
    #     print(sql)
    #     print()
    # exit()
    # print(generated_sqls)

    return generated_sqls
random.seed(42)

def extract_large_numbers(text):
    number_information = []
    patterns = {
        'thousand': 10**3,
        'million': 10**6,
        'billion': 10**9,
        'trillion': 10**12
    }
    
    for word, multiplier in patterns.items():
        matches = re.findall(r'(\d+\.?\d*)\s*{}'.format(word), text, flags=re.IGNORECASE)
        for match in matches:
            number = float(match) * multiplier
            number_information.append(match + " " + word + " = " + str(int(number)))
    
    for phrase, number in {'thousands of': 10**3, 'millions of': 10**6, 'billions of': 10**9, 'trillions of': 10**12}.items():
        if phrase in text:
            number_information.append(phrase + " = " + str(int(number)))
    
    large_number_evidence = ""
    for info in number_information:
        large_number_evidence += info + "; "
    
    return large_number_evidence.strip()

def remove_table_alias(s):
    try:
        tables_aliases = Parser(s).tables_aliases
    except Exception as e:
        return s

    new_tables_aliases = {}
    for i in range(1,11):
        if "t{}".format(i) in tables_aliases.keys():
            new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
    
    tables_aliases = new_tables_aliases
    for k, v in tables_aliases.items():
        # remove AS clauses
        s = s.replace("AS " + k + " ", "")
        # replace table alias with thier original names
        s = s.replace(k, v)
    
    return s

def remove_similar_comments(names, comments):
    '''
    Remove table (or column) comments that have a high degree of similarity with their names
    
    Arguments:
        names: a list of table (or column) names
        comments: a list of table (or column) comments
    
    Returns:
        new_comments: a list of new table (or column) comments
    '''
    new_comments = []
    for name, comment in zip(names, comments):    
        if name.replace("_", "").replace(" ", "") == comment.replace("_", "").replace(" ", ""):
            new_comments.append("")
        else:
            new_comments.append(comment)
    
    return new_comments

def str_replace_ignore_case(evidence, schema_item_name):
    evidence = re.sub(re.escape(schema_item_name), schema_item_name, evidence, 0, re.IGNORECASE)

    return evidence

def obtain_n_grams(sequence, max_n):
    '''
    returns all grams of sequence less than or equal to `max_n`
    '''
    tokens = word_tokenize(sequence)
    all_grams = []
    for n in range(1, max_n + 1):
        all_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    
    return all_grams

def preprocess_evidence(evidence, schema_items):
    if evidence.strip() == "":
        return ""

    evidence = evidence.strip()
    # if evidence does not end with ";", add a ";" char
    if not evidence.endswith(";"):
        evidence += ";"
    
    # lowercase schema items appeared in the evidence
    for table in schema_items:
        if table["table_name"] in evidence.lower():
            evidence = str_replace_ignore_case(evidence, table["table_name"])

        for column_name in table["column_names"]:
            if column_name in evidence.lower():
                evidence = str_replace_ignore_case(evidence, column_name)
    
    evidence = evidence.replace("< =", "<=").replace("> =", ">=")

    return evidence
def prepare_text2sql_prefix_sequence(data):
    prefix_seq = data["schema_sequence"] + "\n" + data["content_sequence"] + "\n" + data["text"] + "\n"
    
    return prefix_seq

def prepare_inputs_and_labels(prefix_seq, target_seq, tokenizer, max_tokens):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq , truncation = False)["input_ids"]
    target_ids = tokenizer(target_seq, truncation = False)["input_ids"] + [tokenizer.eos_token_id]

    seq_length = len(prefix_ids) + len(target_ids)
    if seq_length <= max_tokens: # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention 
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else: # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens-1):]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labels
        labels = labels[-max_tokens:]
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64), 
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64), 
        "labels": torch.tensor(labels, dtype = torch.int64)
    }

def prepare_inputs(prefix_seq, tokenizer, max_prefix_length):
    input_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq , truncation = False)["input_ids"]

    if len(input_ids) > max_prefix_length:
        print("HERE INPUT TOO LONG")
        print("prefix_sequence: ", prefix_seq)
        exit()
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length-1):]
    
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype = torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype = torch.int64)
    }

def create_and_run_conv_dataset(
    dataset_path: str, 
    db_path: str, 
    db_content_index_path: str, 
    source: str, 
    table_json_path: str,
    use_evidence: bool,
    model, # type: ignore
    tokenizer, # type: ignore
    sic_path: str,
    table_num: int,
    column_num: int,
    max_tokens: int,
) -> "list[str]":
    '''
    Load spider-style dataset and run inference with the provided model and tokenizer.
    '''

    dataset = json.load(open(dataset_path))
    additional_db_info = json.load(open(table_json_path))

    db_comments = dict()
    # record comments for tables and columns
    for db_info in additional_db_info:
        comment_dict = dict()

        column_names = [column_name.lower() for _, column_name in db_info["column_names_original"]]
        table_idx_of_each_column = [t_idx for t_idx, _ in db_info["column_names_original"]]
        column_comments = [column_comment.lower() for _, column_comment in db_info["column_names"]]
        
        assert len(column_names) == len(column_comments)
        column_comments = remove_similar_comments(column_names, column_comments)

        table_names = [table_name.lower() for table_name in db_info["table_names_original"]]
        table_comments = [table_comment.lower() for table_comment in db_info["table_names"]]
        
        assert len(table_names) == len(table_comments)
        table_comments = remove_similar_comments(table_names, table_comments)

        # enumerate each table and its columns
        for table_idx, (table_name, table_comment) in enumerate(zip(table_names, table_comments)):
            comment_dict[table_name] = {
                "table_comment": table_comment,
                "column_comments": dict()
            }
            for t_idx, column_name, column_comment in zip(table_idx_of_each_column, column_names, column_comments):
                # record columns in current table
                if t_idx == table_idx:
                    comment_dict[table_name]["column_comments"][column_name] = column_comment

        db_comments[db_info["db_id"]] = comment_dict

    db_ids = set([data["database_id"] for data in dataset])
    db_id2searcher = dict()
    for db_id in db_ids:
        db_id2searcher[db_id] = LuceneSearcher(os.path.join(db_content_index_path, db_id))

    db_id2schema = dict()
    count = 0
    predicted_sqls = []
    new_dataset = []
    # print(source)
    # exit()
    if "cosql" in source or "sparc" in source: # preprocess cosql/sparc to make each interaction split into individual samples
        for data in tqdm(dataset):
            count += 1

            db_id = data["database_id"]
            history = []
            for q in data['interaction']:
                sample = {}
                sample["db_id"] = db_id
                sample['question'] = ""
                if history:
                    sample['question'] += '\n'.join(history)+"\n"
                    sample["question"] += q['utterance']
                else:
                    sample["question"] += q['utterance']
                sample["query"] = q['query']

                # for i in range(3000):
                #     sample['question'] += "\nfiller text"

                history.append(q['utterance'])
                # history.append(q['query'])
                print('Question: \n',sample['question'])
                # exit()

                sample["db_path"] = os.path.join(db_path, db_id, db_id + ".sqlite")
                if db_id in db_id2schema:
                    sample["schema"] = db_id2schema[db_id]
                else:
                    db_id2schema[db_id] = get_db_schema(sample["db_path"], db_comments, db_id)
                    sample["schema"] = db_id2schema[db_id]
                sample["evidence"] = ""
                if "\n" in sample["question"]:
                    sample["question"] = sample["question"].replace("\n", " ")
                if "\n" in sample["evidence"]:
                    sample["evidence"] = sample["evidence"].replace("\n", " ")
                sample["text"] = sample["evidence"] + " " + sample["question"] \
                    if use_evidence and sample["evidence"] != "" else sample["question"]
                sample["sql"] = ""
                sample["table_labels"], sample["column_labels"] = [], []
                try:
                    sql_tokens = [token.value for token in Parser(sample["sql"].lower()).tokens]
                except Exception as e:
                    sql_tokens = sample["sql"].lower().split()
                
                for table_info in sample["schema"]["schema_items"]:
                    sample["table_labels"].append(0)
                    sample["column_labels"].append([0 for _ in range(len(table_info["column_names"]))])
                # coarse-grained matching between the input text and all contents in database
                grams = obtain_n_grams(sample["text"], 4)
                hits = []
                searcher = db_id2searcher[db_id]
                for query in grams:
                    hits.extend(searcher.search(query, k = 10))
                
                # hits = searcher.search(sample["text"], k = 50)

                coarse_matched_contents = dict()
                for i in range(len(hits)):
                    matched_result = json.loads(hits[i].raw)
                    # `tc_name` refers to column names like `table_name.column_name`, e.g., document_drafts.document_id
                    tc_name = ".".join(matched_result["id"].split("-**-")[:2])
                    if tc_name in coarse_matched_contents.keys():
                        if matched_result["contents"] not in coarse_matched_contents[tc_name]:
                            coarse_matched_contents[tc_name].append(matched_result["contents"])
                    else:
                        coarse_matched_contents[tc_name] = [matched_result["contents"]]
                
                fine_matched_contents = dict()
                for tc_name, contents in coarse_matched_contents.items():
                    # fine-grained matching between the question and coarse matched contents
                    fm_contents = get_matched_entries(sample["text"], contents)
                    
                    if fm_contents is None:
                        continue
                    for _match_str, (field_value, _s_match_str, match_score, s_match_score, _match_size,) in fm_contents:
                        if match_score < 0.9:
                            continue
                        if tc_name in fine_matched_contents.keys():
                            if len(fine_matched_contents[tc_name]) < 25:
                                fine_matched_contents[tc_name].append(field_value.strip())
                        else:
                            fine_matched_contents[tc_name] = [field_value.strip()]

                sample["matched_contents"] = fine_matched_contents
                sample["source"] = source
                sic = SchemaItemClassifierInference(sic_path)
                sample = filter_schema_sample(sample, "eval", sic, table_num, column_num)
                del sic
                torch.cuda.empty_cache()
                sample["schema_sequence"] = get_db_schema_sequence(sample["schema"])
                sample["content_sequence"] = get_matched_content_sequence(sample["matched_contents"])
                prefix_seq = prepare_text2sql_prefix_sequence(sample)
                inputs = prepare_inputs(prefix_seq, tokenizer, max_tokens)
                batch_data = inputs
                batch_data["input_ids"] = batch_data["input_ids"].unsqueeze(0)
                batch_data["attention_mask"] = batch_data["attention_mask"].unsqueeze(0)
                for key in batch_data:
                    batch_data[key] = batch_data[key].to(model.device)
                # print('batch data',batch_data)
                # exit()

                generated_sqls = text2sql_func(model, batch_data, tokenizer, max_new_tokens)
                generated_sqls = [post_process(generated_sql, sample["schema"]["schema_items"]) for generated_sql in generated_sqls]

                final_generated_sql = None
                for generated_sql in generated_sqls:
                    execution_error = check_sql_executability(generated_sql, sample["db_path"])
                    if execution_error is None: # the generated sql has no execution errors, we will return it as the final generated sql
                        final_generated_sql = generated_sql
                        break

                if final_generated_sql is None:
                    print("No executable SQL generated.")
                    print("Generated SQLs:")
                    for i, generated_sql in enumerate(generated_sqls):
                        print("SQL {}: {}".format(i, generated_sql))
                    if generated_sqls[0].strip() != "":
                        final_generated_sql = generated_sqls[0]
                    else:
                        final_generated_sql = "SQL placeholder"
                
                print("CHOICE:",final_generated_sql)
                predicted_sqls.append(final_generated_sql)
                history.append(final_generated_sql)
            predicted_sqls.append("")
            # if count > 2:
            #     break
    return predicted_sqls
def filter_schema_sample(sample, sample_type, sic, num_top_k_tables = 5, num_top_k_columns = 5):
    filtered_schema = dict()
    filtered_matched_contents = dict()
    filtered_schema["schema_items"] = []
    filtered_schema["foreign_keys"] = []

    table_names = [table["table_name"] for table in sample["schema"]["schema_items"]]
    table_comments = [table["table_comment"] for table in sample["schema"]["schema_items"]]
    column_names = [table["column_names"] for table in sample["schema"]["schema_items"]]
    column_types = [table["column_types"] for table in sample["schema"]["schema_items"]]
    column_comments = [table["column_comments"] for table in sample["schema"]["schema_items"]]
    column_contents = [table["column_contents"] for table in sample["schema"]["schema_items"]]
    pk_indicators = [table["pk_indicators"] for table in sample["schema"]["schema_items"]]

    # predict scores for each tables and columns
    pred_results = sic.predict(sample)
    # remain top_k1 tables for each database and top_k2 columns for each remained table
    table_probs = [pred_result["table_prob"] for pred_result in pred_results]
    table_indices = np.argsort(-np.array(table_probs), kind="stable")[:num_top_k_tables].tolist()

    for table_idx in table_indices:
        column_probs = pred_results[table_idx]["column_probs"]
        column_indices = np.argsort(-np.array(column_probs), kind="stable")[:num_top_k_columns].tolist()

        filtered_schema["schema_items"].append(
            {
                "table_name": table_names[table_idx],
                "table_comment": table_comments[table_idx],
                "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
            }
        )
    
        # extract matched contents of remained columns
        for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
            tc_name = "{}.{}".format(table_names[table_idx], column_name)
            if tc_name in sample["matched_contents"]:
                filtered_matched_contents[tc_name] = sample["matched_contents"][tc_name]
    
    # extract foreign keys among remianed tables
    filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
    for foreign_key in sample["schema"]["foreign_keys"]:
        source_table, source_column, target_table, target_column = foreign_key
        if source_table in filtered_table_names and target_table in filtered_table_names:
            filtered_schema["foreign_keys"].append(foreign_key)

    # replace the old schema with the filtered schema
    sample["schema"] = filtered_schema
    # replace the old matched contents with the filtered matched contents
    sample["matched_contents"] = filtered_matched_contents

    return sample
if __name__ == "__main__":

    opt = parse_option()
    print(opt)
    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path)
    model = AutoModelForCausalLM.from_pretrained(opt.llm_path, device_map = "auto", torch_dtype = torch.float16)

    # CoSQL dev set (x examples)
    predicted_sqls = create_and_run_conv_dataset(
        dataset_path = "./data/sft_data_collections/cosql/sql_state_tracking/dev.json", 
        db_path = "./data/sft_data_collections/cosql/database", 
        db_content_index_path = "./data/sft_data_collections/cosql/db_contents_index",
        source = "cosql-dev",
        table_json_path = "./data/sft_data_collections/cosql/tables.json",
        use_evidence = False,
        model = model,
        tokenizer = tokenizer,
        sic_path = opt.sic_path,
        table_num = opt.table_num,
        column_num = opt.column_num,
        max_tokens = max_tokens,

    )

    with open(opt.output_path, "w", encoding = 'utf-8') as f:
        for sql in predicted_sqls:
            f.write(sql + "\n")