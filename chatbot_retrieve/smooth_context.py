from copy import deepcopy
from underthesea import sent_tokenize

def extract_consecutive_subarray(numbers):
    subarrays = []
    current_subarray = []
    for num in numbers:
        if not current_subarray or num == current_subarray[-1] + 1:
            current_subarray.append(num)
        else:
            subarrays.append(current_subarray)
            current_subarray = [num]
    subarrays.append(current_subarray)  # Append the last subarray
    return subarrays

def merge_contexts(passages):
    passages_sorted_by_id = sorted(passages, key=lambda x: x["id"], reverse=False)
    psg_ids = [x["id"] for x in passages_sorted_by_id]
    consecutive_ids = extract_consecutive_subarray(psg_ids)

    merged_contexts = []
    b = 0
    for ids in consecutive_ids:
        psgs = passages_sorted_by_id[b:b+len(ids)]
        psg_texts = [x["passage"].strip("Title: ").strip(x["title"]).strip() for x in psgs]
        merged = f"Title: {psgs[0]['title']}\n\n" + " ".join(psg_texts)
        b = b + len(ids)
        merged_contexts.append(dict(
            title=psgs[0]['title'], 
            passage=merged,
            score=max([x["combined_score"] for x in psgs]),
            merged_from_ids=ids
        ))
    return merged_contexts

def discard_contexts(passages):
    sorted_passages = sorted(passages, key=lambda x: x["score"], reverse=False)
    if len(sorted_passages) == 1:
        return sorted_passages
    else:
        shortened = deepcopy(sorted_passages)
        for i in range(len(sorted_passages) - 1):
            current, next = sorted_passages[i], sorted_passages[i+1]
            if next["score"] - current["score"] >= 0.05:
                shortened = sorted_passages[i+1:]
        return shortened

def expand_context(passage, meta_corpus, word_window=60, n_sent=3):
    merged_from_ids = passage["merged_from_ids"]
    title = passage["title"]
    prev_id = merged_from_ids[0] - 1
    next_id = merged_from_ids[-1] + 1
    strip_title = lambda x: x["passage"].strip(f"Title: {x['title']}\n\n")
    
    texts = []
    if prev_id in range(0, len(meta_corpus)):
        prev_psg = meta_corpus[prev_id]
        if prev_psg["title"] == title: 
            prev_text = strip_title(prev_psg)
            prev_text = " ".join(sent_tokenize(prev_text)[-n_sent:])
            texts.append(prev_text)
            
    texts.append(strip_title(passage))
    
    if next_id in range(0, len(meta_corpus)):
        next_psg = meta_corpus[next_id]
        if next_psg["title"] == title: 
            next_text = strip_title(next_psg)
            next_text = " ".join(sent_tokenize(next_text)[:n_sent])
            texts.append(next_text)

    expanded_text = " ".join(texts)
    expanded_text = f"Title: {title}\n{expanded_text}"
    new_passage = deepcopy(passage)
    new_passage["passage"] = expanded_text
    return new_passage

def expand_contexts(passages, meta_corpus, word_window=60, n_sent=3):
    new_passages = [expand_context(passage, meta_corpus, word_window, n_sent) for passage in passages]
    return new_passages

def collapse(passages):
    new_passages = deepcopy(passages)
    titles = {}
    for passage in new_passages:
        title = passage["title"]
        if not titles.get(title):
            titles[title] = [passage]
        else:
            titles[title].append(passage)
    best_passages = []
    for k, v in titles.items():
        best_passage = max(v, key=lambda x: x["score"])
        best_passages.append(best_passage)
    return best_passages

def smooth_contexts(passages, meta_corpus):
    merged_contexts = merge_contexts(passages)
    shortlisted_contexts = discard_contexts(merged_contexts)
    expanded_contexts = expand_contexts(shortlisted_contexts, meta_corpus)
    collapsed_contexts = collapse(expanded_contexts)
    return collapsed_contexts
