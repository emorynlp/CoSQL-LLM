import json
import re
import random

from nltk.tokenize import word_tokenize # type: ignore
from nltk import ngrams # type: ignore
from sql_metadata import Parser # type: ignore
from tqdm import tqdm
random.seed(42)

def extract_large_numbers(text: str) -> str:
    number_information: list[str] = []
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

def remove_table_alias(s: str) -> str:
    try:
        tables_aliases = Parser(s).tables_aliases
    except Exception:
        return s

    new_tables_aliases: dict[str, str] = {}
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

def remove_similar_comments(names: "list[str]", comments: "list[str]") -> "list[str]":
    '''
    Remove table (or column) comments that have a high degree of similarity with their names
    
    Arguments:
        names: a list of table (or column) names
        comments: a list of table (or column) comments
    
    Returns:
        new_comments: a list of new table (or column) comments
    '''
    new_comments: list[str] = []
    for name, comment in zip(names, comments):    
        if name.replace("_", "").replace(" ", "") == comment.replace("_", "").replace(" ", ""):
            new_comments.append("")
        else:
            new_comments.append(comment)
    
    return new_comments

def str_replace_ignore_case(evidence: str, schema_item_name: str) -> str:
    evidence = re.sub(re.escape(schema_item_name), schema_item_name, evidence, 0, re.IGNORECASE)

    return evidence

def obtain_n_grams(sequence: str, max_n: int) -> "list[str]":
    '''
    returns all grams of sequence less than or equal to `max_n`
    '''
    tokens = word_tokenize(sequence)
    all_grams: list[str] = []
    for n in range(1, max_n + 1):
        all_grams.extend([" ".join(gram) for gram in ngrams(tokens, n)])
    
    return all_grams

def preprocess_evidence(evidence: str, schema_items: "list[dict[str, str]]") -> str:
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

def spider_style_dataset(
    dataset_path: str,
    db_path: str, 
    db_content_index_path: str, 
    source: str, 
    table_json_path: str,
    use_evidence: bool,
    mode: str
) -> "list[dict[str, str | None]]":
    '''
    Load spider-style dataset
    
    Arguments:
        dataset_path: directory to load the dataset from
        db_path: directory of databases (used for extracting schema, including tables, columns, column contents, and foreign keys)
        db_content_index_path: directory of database content sparse index
        source: source of examples
        table_json_path: directory to load additional database information (used for extracting comments for tables and columns)
        use_evidence: whether to use the additional evidence in the input sequence
    Returns:
        returned_dataset: prepared dataset
    '''
    returned_dataset = []

    dataset = json.load(open(dataset_path))
    additional_db_info = json.load(open(table_json_path))

    db_comments: "dict[str,str | dict[str, dict[str, str | dict[str, str]]]]" = dict()
    # record comments for tables and columns
    for db_info in additional_db_info:
        comment_dict: "dict[str,dict[str,str|dict[str,str]]]" = dict()

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
                    comment_dict[table_name]["column_comments"][column_name] = column_comment # type: ignore

        db_comments[db_info["db_id"]] = comment_dict

    if "cosql" in source or "sparc" in source: # preprocess cosql to make each interaction split into individual samples
        new_dataset: "list[dict[str, str | None]]" = []
        for data in tqdm(dataset):

            db_id = data["database_id"]
            history: "list[str]" = []
            for q in data['interaction']:
                sample: "dict[str, str | None]" = {}
                sample["db_id"] = db_id
                sample['question'] = ""
                if history:
                    # generate GPT rewritten question
                    sample['question'] += '\n'.join(history)+"\n"
                    sample["question"] += q['utterance']
                    from GQR import GQR
                    sample['question'] = GQR(sample['question'])
                else:
                    sample["question"] += q['utterance']
                sample["query"] = q['query']
                history.append(q['utterance'])
                history.append(q['query'])
                new_dataset.append(sample)
        dataset = new_dataset
    
    returned_dataset = dataset

    return returned_dataset

if __name__ == "__main__":
    print("preparing training sets.....")
    
    
    
    print("cosql-train")
    cosql_train = []
    # CoSQL training set (x examples)
    cosql_train = spider_style_dataset(
        dataset_path = "./data/sft_data_collections/cosql/sql_state_tracking/train.json",
        db_path = "./data/sft_data_collections/cosql/database", 
        db_content_index_path = "./data/sft_data_collections/cosql/db_contents_index",
        source = "cosql-train",
        table_json_path = "./data/sft_data_collections/cosql/tables.json",
        use_evidence = False,
        mode = "train"
    )
    print(len(cosql_train))
    with open("./gpt/sft_cosql_train.json", "w") as f:
        f.write(json.dumps(cosql_train, indent = 2, ensure_ascii = False))


    print("sparc-train")
    sparc_train = []
    # sparc training set (x examples)
    sparc_train = spider_style_dataset(
        dataset_path = "./data/sft_data_collections/sparc/train.json",
        db_path = "./data/sft_data_collections/sparc/database", 
        db_content_index_path = "./data/sft_data_collections/sparc/db_contents_index",
        source = "sparc-train",
        table_json_path = "./data/sft_data_collections/sparc/tables.json",
        use_evidence = False,
        mode = "train"
    )
    print(len(sparc_train))
    with open("./gpt/sft_sparc_train.json", "w") as f:
        f.write(json.dumps(sparc_train, indent = 2, ensure_ascii = False))


    print("---------------------------------------------------------------------------")
    print("preparing dev sets.....")
    

    print("cosql-dev")
    # CoSQL dev set (x examples)
    cosql_dev = spider_style_dataset(
        dataset_path = "./data/sft_data_collections/cosql/sql_state_tracking/dev.json", 
        db_path = "./data/sft_data_collections/cosql/database", 
        db_content_index_path = "./data/sft_data_collections/cosql/db_contents_index",
        source = "cosql-dev",
        table_json_path = "./data/sft_data_collections/cosql/tables.json",
        use_evidence = False,
        mode = "dev"
    )
    print(len(cosql_dev))
    with open("./gpt/sft_cosql_dev.json", "w") as f:
        f.write(json.dumps(cosql_dev, indent = 2, ensure_ascii = False))


    print("sparc-dev")
    # sparc dev set (x examples)
    sparc_dev = spider_style_dataset(
        dataset_path = "./data/sft_data_collections/sparc/dev.json", 
        db_path = "./data/sft_data_collections/sparc/database", 
        db_content_index_path = "./data/sft_data_collections/sparc/db_contents_index",
        source = "sparc-dev",
        table_json_path = "./data/sft_data_collections/sparc/tables.json",
        use_evidence = False,
        mode = "dev"
    )
    print(len(sparc_dev))
    with open("./gpt/sft_sparc_dev.json", "w") as f:
        f.write(json.dumps(sparc_dev, indent = 2, ensure_ascii = False))