from itertools import chain
import json
import os

from FlagEmbedding import FlagModel, FlagReranker
import jieba
from llama_index.readers.file import PyMuPDFReader
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define chunking strategies based on category
def chunk(texts: list[str], category: str):
    """
    對文本進行分塊。
    
    參數：
        texts(list[str]): 原始文本
        category(str): 文本類別
           
    回傳值:
        list[str]: 分塊後之結果
    """
    if category == 'insurance':
        chunk_size = 200
        chunk_overlap = 0
    elif category == 'finance':
        chunk_size = 800
        chunk_overlap = 400
    elif category == 'faq':
        chunk_size = 400
        chunk_overlap = 200
    else:
        chunk_size = 400
        chunk_overlap = 200 # Default settings

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",
            "\uff0c",
            "\u3001",
            "\uff0e",
            "\u3002",
            "",
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = []
    for text in texts:
        chunks.extend(text_splitter.split_text(text))
    return chunks

class DocumentIndex:
    """
    用以儲存與管理文本之資料結構
    
    屬性:
        docs(list[str]): 所有文本頁面之列表
        doc_page_ids(dict[int, list[int]]): 從文件ID到頁面ID列表之映射表
        doc_ids(list[int]): 從頁面ID到文件ID之映射表
        emdeds: 從頁面ID到嵌入向量之映射表
    """
    def __init__(self, docs, doc_page_ids, doc_ids):
        self.docs: list[str] = docs
        self.doc_page_ids: dict[int, list[int]] = doc_page_ids
        self.doc_ids: list[int] = doc_ids
        self.embeds = []

    def get_pages(self, doc_id_list: list[int]):
        """
        此函數蒐集並回傳指定文件包含的所有頁面ID。
        
        參數：
            doc_id_list(list[int]): 欲查詢之文件ID列表
           
        回傳值:
            list[int]: 頁面ID之列表
        """
        return list(chain(*[self.doc_page_ids[i] for i in doc_id_list]))

    def select_embeds(self, page_id_list: list[int]):
        """
        此函數蒐集並回傳指定頁面文本之嵌入向量。
        
        參數：
            page_id_list(list[int]): 欲查詢之頁面ID列表
           
        回傳值:
            嵌入向量之列表
        """
        return [self.embeds[i] for i in page_id_list]

def load_pdfs(source_path, category):
    """
    載入PDF文件。
    
    參數：
        source_path: 檔案路徑
           
    回傳值:
        list[DocumentIndex]: PDF文本
    """
    loader = PyMuPDFReader()
    docs = []
    doc_page_ids = {}
    doc_ids = []

    for file in tqdm(os.listdir(source_path)):
        idx = int(file.replace('.pdf', ''))  
        pages = [doc.text for doc in loader.load_data(os.path.join(source_path, file))]
        splitted_docs = chunk(pages, category)

        doc_page_ids[idx] = list(range(len(docs), len(docs) + len(splitted_docs)))
        docs.extend(splitted_docs)
        doc_ids.extend([idx] * len(splitted_docs))

    return DocumentIndex(docs, doc_page_ids, doc_ids)

def get_ranking(scores):
    """
    計算排名。
    
    參數：
        scores: 第1~N個對象之分數
           
    回傳值:
        np.ndarray: 按照分數計算之排名。
    """
    return scores.argsort()[::-1].argsort() + 1

def BM25_scores(qs, source, corpus_dict):
    """
    計算BM25分數。
    
    參數：
        qs: 查詢語句
        source: 參考文本之ID
        corpus_dict: 文本字典 
           
    回傳值:
        BM25演算法計算出之分數
    """
    filtered_corpus = [corpus_dict[int(id)] for id in source]
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = list(jieba.cut_for_search(qs))
    return bm25.get_scores(tokenized_query)

if __name__ == '__main__':
    mode = 'rerank'
    model_name = 'BAAI/bge-reranker-large'

    if mode == 'rerank':
        rerank_model = FlagReranker(model_name, use_fp16=True)
    elif mode == 'embed':
        embed_model = FlagModel(model_name, use_fp16=True)
    else:
        raise ValueError('mode must be "embed" or "rerank"')

    dim = 1024
    alpha = 0.5
    score_method = 'ws'

    doc_index_ins = load_pdfs('reference/insurance', 'insurance')
    doc_index_fin = load_pdfs('reference/finance', 'finance')

    with open(os.path.join('reference/faq/pid_map_content.json'), 'r', encoding='utf8') as f_s:
        key_to_source_dict = json.load(f_s)
        docs = []
        doc_page_ids = {}
        doc_ids = []
        for k, v in key_to_source_dict.items():
            idx = int(k)
            chunks = chunk([str(v)], 'faq')
            doc_page_ids[idx] = list(range(len(docs), len(docs) + len(chunks)))
            docs.extend(chunks)
            doc_ids.extend([idx] * len(chunks))
        doc_index_faq = DocumentIndex(docs, doc_page_ids, doc_ids)

    if mode == 'embed':
        doc_index_ins.embeds = embed_model.encode(doc_index_ins.docs)
        doc_index_fin.embeds = embed_model.encode(doc_index_fin.docs)
        doc_index_faq.embeds = embed_model.encode(doc_index_faq.docs)

    doc_index_dict = {'insurance': doc_index_ins, 'finance': doc_index_fin, 'faq': doc_index_faq}

    answer_dict = {"answers": []}

    with open('questions_preliminary.json', 'r', encoding='utf8') as f:
        qs_ref = json.load(f)

    for q_dict in tqdm(qs_ref['questions']):
        category = q_dict['category']
        query = [q_dict['query']]
        sources = q_dict['source']
        doc_index = doc_index_dict[category]
        candidates = doc_index.get_pages(sources)

        if mode == 'rerank':
            dense_scores = np.array(
                [rerank_model.compute_score([query[0], doc_index.docs[i]]) for i in candidates]
            ).squeeze()
        elif mode == 'embed':
            query_vector = embed_model.encode_queries(query, batch_size=1)
            candidate_embeds = doc_index.select_embeds(candidates)
            dense_scores = cosine_similarity(query_vector, candidate_embeds)[0]

        sparse_score = BM25_scores(query[0], candidates, doc_index.docs)

        if score_method == 'rrf':
            dense_ranking = get_ranking(dense_scores)
            sparse_ranking = get_ranking(sparse_score)
            scores = 1 / (60 + dense_ranking) + 1 / (60 + sparse_ranking)
        elif score_method == 'ws':
            sparse_score = (sparse_score - sparse_score.min()) / (sparse_score.max() - sparse_score.min() + 1e-8)
            scores = alpha * dense_scores + (1 - alpha) * sparse_score

        if scores.max() != scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        candidate_scores = sorted(
            [{"doc_id": doc_index.doc_ids[candidates[i]], "score": float(scores[i])} for i in range(len(candidates))],
            key=lambda x: x["score"],
            reverse=True
        )[:3]

        best_match = candidate_scores[0]["doc_id"]
        answer_dict['answers'].append({
            "qid": q_dict['qid'],
            "retrieve": best_match,
            "top_3_scores": candidate_scores
        })

    output_path = f'answer_chunk_{mode}_{model_name.replace("/", "_")}_{score_method}.json'
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
