import os
import time
import urllib3
import sys
import faiss
import json
import transformers
import warnings
import threading
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tqdm as tqdm
from collections import defaultdict
from transformers import AutoModel
import transformers
from utils.tool_utils import *
import nltk
from more_itertools import chunked
import numpy as np
import pickle
from typing import Dict, List, Union, Tuple, Any
from utils.utils import read_plain_csv

from utils.tool_utils import Embedder, Reranker, excel_to_markdown

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
device = "cuda:0"

class SemanticRetriever :
    """
    Retrieving process, containing recall and rerank.
    """
    def __init__(
        self,
        chunks: List[str],
        chunk_index: Dict,
        chunk_file_index: Dict = None,
        embedder: Embedder=None,
        reranker_path: str = None,
        save_path: str = "./retrieval_result/embedding.pkl"
    ) -> None:
        self.embedding_model = embedder
        self.reranker = Reranker(reranker_path)

        if os.path.exists(save_path) :
            doc_embeddings, self.chunks, self.chunk_file_index = self.load_embeddings(save_path)
            self.chunk_index = {idx: ch for idx, ch in enumerate(self.chunks)}
        else :
            self.chunks = chunks
            self.chunk_index = chunk_index
            self.chunk_file_index = chunk_file_index
            doc_embeddings = self.embed_doc(chunks, save_path=save_path)

        self.thread_local = threading.local()
        self.index_lock = threading.RLock()

        print("embedding size", doc_embeddings.shape)
        self.res = faiss.StandardGpuResources()
        self.index_IP = self.build_index(doc_embeddings)

    def embed_doc(self, chunks: List[str], batch_size: int = 512, save_path: str = None) -> Any :
        """
        Embed documents in batches for improved performance

        Args:
            chunks: List of text chunks to embed

        Returns:
            np.ndarray: Array of document embeddings
        """
        encode_vecs = []
        iterator = tqdm(range(0, len(chunks), batch_size)) if len(chunks) >= 100 \
            else range(0, len(chunks), batch_size)

        for i in iterator :
            batch = chunks[i: i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch)

            if len(batch) == 1 :
                encode_vecs.append(batch_embeddings)
            else :
                encode_vecs.extend(batch_embeddings)

        encode_vecs = np.array(encode_vecs)
        if len(encode_vecs.shape) == 3 :
            encode_vecs = encode_vecs.reshape(-1, encode_vecs.shape[-1])

        if save_path :
            self.save_embeddings(encode_vecs, chunks, save_path)
            print("Embedding Vectors Saved.")

        return encode_vecs

    def save_embeddings(self, embeddings: Any, chunks: List[str], save_path: str) -> None :
        """
        Save embeddings and optionally the original chunks.

        Args:
            embeddings: numpy array of embeddings
            chunks: orignal text chunks
            save_path: path to save the embeddings
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith('.pkl') :
            data = {
                "embeddings": embeddings,
                "chunks": chunks,
                "chunk_file_index": self.chunk_file_index
            }
            with open(save_path, "wb") as f :
                pickle.dump(data, f)
            print(f"Embeddings and chunks saved to {save_path}")

    @staticmethod
    def load_embeddings(load_path: str = None) -> Tuple[Any, Any] :
        """
        Load saved embeddings.

        Args:
            load_path: Path to the saved embeddings.
        
        Returns:
            embeddings if .npy file, (embeddings, chunks) if .pkl file
        """
        if load_path.endswith('.npy') :
            return np.load(load_path)
        elif load_path.endswith('.pkl') :
            with open(load_path, 'rb') as f :
                data = pickle.load(f)
            return data['embeddings'], data['chunks'], data['chunk_file_index']

    def build_index(self, dense_vector: Any) -> Any :
        print("Building Index.")
        with self.index_lock :
            _, dim = dense_vector.shape
            index_IP = faiss.IndexFlatIP(dim)
            co = faiss.GpuClonerOptions()

            # make it to gpu index
            # index_gpu = faiss.index_cpu_to_gpu(provider=self.res, device=2, index=index_IP, options=co)
            index_gpu = index_IP
            index_gpu.add(dense_vector)

            return index_gpu

    def retrieve(self, query, recall_num, rerank_num) :
        """
        调用 self.recall，从向量索引里取出与 query 最相似的 recall_num 个文档片段 docs，以及它们对应的原始文件名 ori_file_name。
        再调用 self.rerank，对这批候选 docs 做精排，只保留 rerank_num 个最相关的：
            reranked_docs：排好序后的文本片段。
            rerank_scores：对应的相关性分数。
            filenames：对应的原始文件名。
        """
        docs, ori_file_name = self.recall(query, recall_num)
        reranked_docs, rerank_scores, filenames = self.rerank(query, docs, rerank_num, ori_file_name)
        return reranked_docs, rerank_scores, filenames

    def recall(self, query: str, topn:int) -> List[str] :
        query_emb = self.embed_doc(query)
        with self.index_lock :
            D, I = self.index_IP.search(query_emb, topn)
        ori_docs = [self.chunk_index[i] for i in I[0]]
        ori_file_name = [self.chunk_file_index[i] for i in I[0]]
        return ori_docs, ori_file_name

    def rerank(self, query: str, docs: List[str], topn: int, ori_file_name: List[str]) -> Tuple[List[str], List[int]] :
        pairs = [[query, d] for d in docs]      # 构造 [query, 文档片段] 的成对输入列表，给交叉编码重排模型用
        scores = self.reranker.compute_score(pairs) # 对每个 [query, doc] 输出一个相关性分数
        sroted_pairs = sorted(zip(scores, docs, ori_file_name), reverse=True) #打包在一起，按分数从高到低排序
        score_sorted, doc_sorted, filename_sorted = zip(*sroted_pairs) # 再拆开成 3 个对齐的列表：分数、文本、文件名
        return doc_sorted[:topn], score_sorted[: topn], filename_sorted[: topn]


class MixedDocRetriever :
    def __init__(
        self,
        doc_dir_path: str,
        excel_dir_path: str,
        embedder: Embedder,
        reranker_path: str = None,
        save_path: str = "./retrieve_result/embedding.pkl"
    ) -> None:
        self.ori_documents = self.load_hybrid_dataset(doc_dir_path, excel_dir_path)
        print("Loading done.")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_chunking_dict = self.doc_chunking()
        self.chunks, self.chunk_to_index, self.chunk_to_filename = self.build_index(doc_chunking_dict)
        self.semantic_retriever = SemanticRetriever(
            chunks=self.chunks,
            chunk_index=self.chunk_to_index,
            chunk_file_index=self.chunk_to_filename,
            embedder=embedder,
            reranker_path=reranker_path,
            save_path=save_path
        )

    def load_hybrid_dataset(self, doc_dir_path: str, excel_dir_path: str) -> Dict[str, List[str]] :
        all_docs = defaultdict(list)
        for file in tqdm(os.listdir(excel_dir_path)) :
            content = excel_to_markdown(os.path.join(excel_dir_path, file))
            excel_content = content
            all_docs[file] = excel_content

        for file in tqdm(os.listdir(doc_dir_path)) :
            with open(os.path.join(doc_dir_path, file), 'r', encoding="utf-8") as fin :
                data_split = json.load(fin)
            key_value_doc = ''
            for key, item in data_split.items() :
                key_value_doc += f"{key} {item}\n"

            all_docs[file] = key_value_doc
        return all_docs


    def build_index(self, chunking_dict: Dict) -> Tuple :
        flatten_chunks = []
        chunk_to_index = defaultdict(list)
        chunk_to_filename = defaultdict(str)
        cnt = 0
        for idx, (key, item) in enumerate(chunking_dict.items()) :
            flatten_chunks += item
            for i in item :
                chunk_to_index[cnt] = i
                chunk_to_filename[cnt] = key
                cnt += 1
        return flatten_chunks, chunk_to_index, chunk_to_filename

    def nltk_single_doc_chunking(self, doc: str, key: str) -> Tuple[List[str], str] :
        all_splits = self.text_splitter.split_text(doc)
        add_file_name_splits = []
        for split_chunk in all_splits :
            add_file_name_chunk = f"File name: {key}\n" + split_chunk
            add_file_name_splits.append(add_file_name_chunk)
        return add_file_name_splits, key

    def doc_chunking(self, max_workers=10) -> Dict :
        doc_chunkings = defaultdict(list)
        with ThreadPoolExecutor(max_workers=max_workers) as executor :
            future_to_case = {executor.submit(self.nltk_single_doc_chunking, doc, key): doc for key, doc in self.ori_documents.items()}
            for future in tqdm(as_completed(future_to_case), total=len(self.ori_documents), desc="Processing cases") :
                try :
                    single_doc_chunking, key = future.result()
                    doc_chunkings[key] += single_doc_chunking
                except Exception as e :
                    print(f"Case processing generated exception: {e}")
        return doc_chunkings

    def retrieve(self, query: str, recall_nun: int = 50, rerank_num: int = 5) :
        return self.semantic_retriever.retrieve(query, recall_nun, rerank_num)





