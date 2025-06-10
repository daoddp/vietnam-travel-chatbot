from datasets import load_dataset

def load_meta_corpus(file_path):
    """
    Load corpus from a JSONL file.
    """
    return load_dataset("json", data_files=file_path, split="train").to_list()
