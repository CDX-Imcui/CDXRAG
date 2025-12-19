import numpy as np
import pandas as pd
import json
import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import math

from chat_utils import get_chat_result
from config import config_mapping
from utils.tool_utils import Embedder
from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    print(f"⏱️  [{name}] Time: {end - start:.4f}s")


class TableRAGPipeline:
    """
    集成了：表格重构、BGE 向量检索、Schema Pruning (列筛选) 和 子表生成。
    """

    def __init__(self,
                 df: pd.DataFrame,
                 external_text_list: List[str],  # 核心改动：直接输入字符串列表
                 llm_backbone: str,
                 embedder: Embedder):

        self.df = df
        self.raw_text_list = external_text_list
        # 1. 加载 LLM 配置
        self.llm_config = config_mapping.get(llm_backbone)
        if not self.llm_config:
            raise ValueError(f"Backbone {llm_backbone} not found in config_mapping")

        # 预处理：转字符串，填充空值
        self.df = self.df.astype(str).replace('nan', '')
        self.embedder = embedder

        # 4. 内部状态存储
        self.documents = []  # 存储转化后的实体文档
        self.table_embeddings = None  # 表格行向量 (Tensor)
        self.text_embeddings = None  # 文本块向量
        self.template = ""  # 存储生成的通用模板
        self.pk_col = self.df.columns[0]  # 默认第一列为主键

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_model_name = "models/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name).to(self.device)
        self.nli_model.eval()  # 务必开启 eval 模式，关闭 Dropout
        self.nli_labels = ["entailment", "neutral", "contradiction"]

    def _clean_json_response(self, content: str) -> Dict:
        """Helper: 鲁棒的 JSON 提取器"""
        content = content.strip()
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        json_str = match.group(1) if match else content
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"❌ JSON Parse Failed. Raw:\n{content}")
            return {}

    # =========================================================================
    # PHASE 1: 离线索引构建 (Offline Indexing)
    # =========================================================================

    def _generate_generic_template(self) -> Dict:
        """让 LLM 看表头，生成一个通用的、中立的行描述模板"""
        columns = self.df.columns.tolist()
        prompt = """
You are a Data-to-Text Template Generator.
Input Columns: {columns}

Goal: Create a python format string to convert a table row into a natural language sentence.

CRITICAL RULES (Follow Strictness Level: MAX):
1. **DO NOT change column names.** Keep them EXACTLY as provided in the Input Columns.
2. **DO NOT replace spaces with underscores.**
   - WRONG: {{Software_license}}
   - CORRECT: {{Software license}}
3. Use double curly braces for placeholders: {{Column Name}}.
4. Do NOT infer or hallucinate information not present in the columns.

Output JSON only:
{{
  "primary_key": "<best identifier column>",
  "template": "<sentence template>"
}}
"""
        formatted_prompt = prompt.format(columns=', '.join(columns))
        print(f"🤖 [LLM] Generating generic row template...")
        response = get_chat_result(
            messages=[{"role": "user", "content": formatted_prompt}],
            tools=None,
            llm_config=self.llm_config
        )
        return self._clean_json_response(response.content)

    def build_index(self):
        """核心流程：执行离线建库"""
        print("\n=== 🏗️ Phase 1: Building Offline Index ===")

        # 1. 生成模板
        template_info = self._generate_generic_template()
        self.template = template_info.get("template", "")
        self.pk_col = template_info.get("primary_key", self.df.columns[0])
        print(f"✅ Template: {self.template}")

        # 2. 行转文本 (Entity Documents)
        py_template = self.template.replace("{{", "{").replace("}}", "}")
        self.documents = []
        table_texts = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Rows to Docs"):
            row_dict = row.to_dict()
            try:
                text = py_template.format(**row_dict)
                self.documents.append({
                    "row_id": idx,
                    "text": text,
                    "entity": row_dict.get(self.pk_col, "Unknown")
                })
                table_texts.append(text)
            except Exception:
                continue

        # 3. BGE 向量化 (Vectorization)
        print("⚡ Encoding with BGE...")
        if not table_texts:
            raise ValueError("❌ No texts generated from table! Check your template keys against dataframe columns.")
        raw_emb = torch.tensor(self.embedder.encode(table_texts))
        # 手动进行 L2 归一化 (p=2, dim=1)
        self.table_embeddings = F.normalize(raw_emb, p=2, dim=1)

        # 对外部文本列表进行向量化
        if self.raw_text_list and len(self.raw_text_list) > 0:
            print(f"⚡ Encoding {len(self.raw_text_list)} External Text Blocks...")
            self.text_embeddings = F.normalize(torch.tensor(self.embedder.encode(self.raw_text_list)), p=2, dim=1)
        else:
            print("⚠️ Warning: external_text_list is empty, text indexing skipped.")

    # =========================================================================
    # 推理
    # =========================================================================

    def _get_top_k_indices(self, query_emb: torch.Tensor, embeddings: torch.Tensor, top_k: int) -> List[int]:
        """统一检索核心：处理 Query 编码与相似度计算"""
        if embeddings is None: return []
        # 计算点积相似度
        scores = torch.matmul(embeddings, query_emb)
        top_results = torch.topk(scores, k=min(top_k, embeddings.shape[0]))
        return top_results.indices.tolist()

    def _filter_columns(self, question: str) -> Dict[str, Any]:
        """让 LLM 根据问题筛选列，并判断是否需要表外知识"""
        all_cols = self.df.columns.tolist()
        prompt = """
You are a Table Column Selector for table question answering.

Input:
- Question: "{question}"
- Available Columns: {columns}

Goal:
Select a MINIMALLY SUFFICIENT set of columns to answer the question using ONLY the table.
"Minimally sufficient" means the chosen columns are enough to:
(A) locate the target row(s),
(B) perform any required operations (filter/sort/rank/aggregate/compare),
(C) extract the final answer value.

Critical constraints:
1) You may ONLY choose from the provided column names and MUST preserve the exact column strings.
2) Always include at least one entity identifier / primary-key-like column (e.g., name/player/id) if such a column exists.
3) If the question involves ranking or "most/second/top", include BOTH:
   - the metric column (e.g., Yards/Score/Count), AND
   - the rank column, unless you are certain rank is derived from exactly that same metric.
4) IMPORTANT: If the final answer is NOT explicitly available in the table columns,
   OR the question requires external descriptive facts,
   set "answer_in_table" to false.
   If the table alone is sufficient, set "answer_in_table" to true.
5) Notes / remarks columns:
   Columns such as "Notes", "Remarks", "Comments", or similar
   should be kept by default if present

Output JSON only:
{{
  "selected_columns": ["<exact column name>", ...],
  "answer_in_table": true/false,
  "reasoning": "<brief explanation>"
}}
    """
        formatted_prompt = prompt.format(question=question, columns=', '.join(all_cols))
        response = get_chat_result(
            messages=[{"role": "user", "content": formatted_prompt}],
            tools=None,
            llm_config=self.llm_config
        )

        result = self._clean_json_response(response.content)

        # 校验选出的列是否真的在表中
        result["selected_columns"] = [c for c in result.get("selected_columns", []) if c in all_cols]
        if not result["selected_columns"]:
            result["selected_columns"] = all_cols

        print(f"🏷️ answer_in_table: {result['answer_in_table']}")
        print(f"💡 Reasoning: {result['reasoning']}")

        return result

    def _analyze_query_intent(self, question: str) -> Dict[str, bool]:
        """
        分析问题意图：是简单的查值，还是复杂的聚合/排序
        """
        signals = {
            "is_complex": False,
            "has_agg": any(w in question.lower() for w in ["how many", "sum", "average", "total", "percentage"]),
            "has_rank": any(w in question.lower() for w in ["most", "highest", "second", "rank", "top", "compare"])
        }
        if signals["has_agg"] or signals["has_rank"]:
            signals["is_complex"] = True
        return signals

    def _expand_context_radius(self, anchor_ids: List[int], intent: Dict[str, bool]) -> List[int]:
        """
        根据意图和分布情况，自适应分配上下文预算
        """
        final_ids = set(anchor_ids)

        # 只要意图是简单的（查值），就强制走 Compact 策略，忽略分布离散度 (std_dist)
        # 只有当问题确实需要聚合/比较 (is_complex=True) 时，才考虑属性扩展
        if not intent["is_complex"]:
            for rid in anchor_ids:
                if rid > 0: final_ids.add(rid - 1)
                if rid < len(self.df) - 1: final_ids.add(rid + 1)

        # 复杂型 (聚合/排序) -> 属性共享扩展
        else:
            for rid in anchor_ids:
                shared_val = self.df.iloc[rid].get('Current layout engine', '')
                if shared_val and shared_val != 'Unknown':
                    shared_rows = self.df[self.df['Current layout engine'] == shared_val].index.tolist()
                    final_ids.update(shared_rows)

        sorted_ids = sorted(list(final_ids))
        return sorted_ids[:15]

    # =========================================================================
    # 文本侧精简 (Textual Pruning - KV Focused)
    # =========================================================================

    def _retrieve_and_prune_text(self, query_emb: torch.Tensor, anchor_entities: List[str],
                                 retrieved_texts: List[str]) -> List[Dict]:
        """
        2. 自动判定 KV 结构与句子结构
        3. 基于 BGE 相似度与实体锚定打分
        4. 动态保留前 50% 的高价值信息单元
        """
        if not retrieved_texts: return []

        entity_keywords = set()
        for ent in anchor_entities:
            for word in re.split(r'\W+', ent):  # 按非字母字符拆分
                if len(word) > 3:  entity_keywords.add(word.lower())

        seen_units = set()  # 用于去重
        for text in retrieved_texts:
            # 自动判定 KV vs 纯文本结构
            is_kv = len(re.findall(r'[:：|]', text)) > len(text) / 50
            units = re.split(r'[\n;]', text) if is_kv else re.split(r'(?<=[。？！?.])\s+', text)
            for u in units:
                u_clean = u.strip()
                if len(u_clean) > 5 and u_clean not in seen_units:
                    seen_units.add(u_clean)

        unique_units = list(seen_units)
        if not seen_units: return []

        # 向量化 (增加手动归一化，确保后续计算准确)
        # raw_embs: [N, Dim]
        raw_embs = torch.tensor(self.embedder.encode(unique_units))
        unit_embs = torch.nn.functional.normalize(raw_embs, p=2, dim=1)
        # 打分 (Query vs Units)
        if query_emb.dim() == 1:
            scores = torch.matmul(unit_embs, query_emb)
        else:
            scores = torch.matmul(unit_embs, query_emb.t()).squeeze()
        scores = scores.cpu().numpy()

        all_units = []
        for i, score in enumerate(scores):
            text_unit = unique_units[i]
            # 关键词加分
            if any(kw in text_unit.lower() for kw in entity_keywords):
                score += 0.2
            all_units.append({
                "text": text_unit,
                "score": score,
                "embedding": unit_embs[i]  # 带出向量，供下一步对齐使用
            })

        # 保留前 50%
        all_units.sort(key=lambda x: x["score"], reverse=True)
        keep_count = min(20, math.ceil(len(all_units) * 0.5))  # 稍微放宽一点上限到20，保证上下文

        return all_units[:keep_count]

    def _inject_cross_references(self, sub_df: pd.DataFrame, pruned_units: List[Dict]) -> Dict[str, str]:
        """
        核心功能：建立双向引用 (Bi-directional Reference)
        1. Table -> Text: 在表格中添加文本 ID 和相似度 (Top-5)。
        2. Text -> Table: 在文本前标记它属于哪些实体 (Multi-label)。
        利用 Pruning 阶段产生的单元向量，计算表格行与文本单元的引用关系。
        """
        if not pruned_units:
            return {"table_md": sub_df.to_markdown(index=False), "text_str": ""}

        # 1. 准备数据
        # 提取单元向量堆叠成矩阵 [M, Dim]
        unit_embs = torch.stack([u['embedding'] for u in pruned_units])

        # 提取子表行向量 [K, Dim]
        row_indices = sub_df.index.tolist()
        row_embs = self.table_embeddings[row_indices]  # 注意：table_embeddings 最好在 build_index 里已经归一化

        # 2. 计算相似度矩阵 [K, M]
        sim_matrix = torch.matmul(row_embs, unit_embs.t())

        # 3. 双向打标容器
        row_refs = {i: [] for i in range(len(sub_df))}  # 表格行 -> 引用ID
        unit_labels = {j: set() for j in range(len(pruned_units))}  # 文本单元 -> 实体名

        # 4. 遍历表格行，寻找匹配的文本单元
        for r_idx in range(len(sub_df)):
            row_entity = str(sub_df.iloc[r_idx][self.pk_col])
            ent_keywords = {w.lower() for w in re.split(r'\W+', row_entity) if len(w) > 3}

            scores = sim_matrix[r_idx]

            # 这里的阈值可以稍低，因为 Pruning 阶段已经筛选过一轮了
            # 找出 Top-5 且分数 > 0.45 的单元
            top_k_indices = torch.nonzero(scores > 0.45).squeeze()
            if top_k_indices.dim() == 0 and top_k_indices.item() is None: continue
            if top_k_indices.dim() == 0:
                top_k_indices = [top_k_indices.item()]
            else:
                top_k_indices = top_k_indices.tolist()

            # 按分数排序取 Top 5
            top_k_pairs = sorted([(scores[i].item(), i) for i in top_k_indices], key=lambda x: x[0], reverse=True)[:5]

            for score, u_idx in top_k_pairs:
                text_content = pruned_units[u_idx]['text']

                # 双重校验：要么分数极高，要么包含实体关键词
                is_keyword_match = any(kw in text_content.lower() for kw in ent_keywords)
                is_high_conf = score > 0.75

                if is_keyword_match or is_high_conf:
                    # 表格侧记录: [0](0.82)
                    row_refs[r_idx].append(f"[{u_idx}]({score:.2f})")
                    # 文本侧记录: Android browser
                    unit_labels[u_idx].add(row_entity)

        # 5. 生成增强版表格
        view_df = sub_df.copy()
        view_df["Related Context IDs"] = [", ".join(refs) for refs in row_refs.values()]
        table_md = view_df.to_markdown(index=False)

        # 6. 生成增强版文本串
        formatted_texts = []
        for i, unit in enumerate(pruned_units):
            labels = sorted(list(unit_labels[i]))
            label_str = f"[Rel: {', '.join(labels)}]" if labels else ""
            # 格式: [0] [Rel: Android] The text content...
            formatted_texts.append(f"[{i}] {label_str} {unit['text']}")

        return {
            "table_md": table_md,
            "text_str": "\n".join(formatted_texts)
        }

    def _verify_evidence(self, sub_table_facts: List[str], text_evidence: str) -> List[str]:
        """
        利用 Tokenizer 的 Batch 处理能力，一次性校验所有表格事实
        """
        if not text_evidence or not sub_table_facts:
            return []

        verification_signals = []
        # 将文本证据作为统一的前提 (Premise)
        premise = text_evidence[:1500]

        try:
            entail_idx = self.nli_labels.index("entailment")
            contra_idx = self.nli_labels.index("contradiction")
        except ValueError:
            # 兜底逻辑：如果 labels 设置不对，默认使用官方标准 0, 2
            entail_idx, contra_idx = 0, 2

        # 1. 构造 Batch 输入对：[[Premise, Hypo1], [Premise, Hypo2], ...]
        pairs = [[premise, fact] for fact in sub_table_facts]

        # 2. 调用 Tokenizer 的批处理功能
        # padding=True 会自动对齐长度，return_tensors="pt" 返回 PyTorch 张量
        inputs = self.nli_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # 3. 开启无梯度推理模式
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            # 对 logits 在最后一个维度（标签维度）做 Softmax，得到概率分布 [Batch_size, 3]
            predictions = torch.softmax(outputs.logits, dim=-1)

        # 4. 解析结果 (对应官方标签顺序: entailment, neutral, contradiction)
        # 将结果转回 CPU 列表处理
        predictions = predictions.cpu().numpy()

        for i, probs in enumerate(predictions):
            fact = sub_table_facts[i]
            entail_prob = probs[entail_idx]
            contra_prob = probs[contra_idx]

            # 阈值判定：只有置信度够高才输出信号，减少噪声
            if entail_prob > 0.7:
                verification_signals.append(f"✅ Fact Verified: {fact[:60]}... (Conf: {entail_prob:.1%})")
            elif contra_prob > 0.7:
                verification_signals.append(f"❌ Conflict Detected: {fact[:60]}... (Conf: {contra_prob:.1%})")

        return verification_signals

    def retrieve_aligned_context(self, question: str):
        """
        推理入口：结合自适应子表与精简 KV 文本
        """
        print(f"\n=== 🚀 Hybrid Query: {question} ===")
        query_emb_numpy = self.embedder.encode(question)
        query_emb = torch.tensor(query_emb_numpy).squeeze()

        # 1. 意图分析与锚点检索
        intent = self._analyze_query_intent(question)
        anchor_ids = self._get_top_k_indices(query_emb, self.table_embeddings, top_k=10)
        anchor_entities = [self.df.iloc[rid][self.pk_col] for rid in anchor_ids]
        expanded_ids = self._expand_context_radius(anchor_ids, intent)

        # 3.  构建精简子表
        col_info = self._filter_columns(question)
        is_sufficient = col_info.get('answer_in_table', False)  # 获取这个关键信号
        # 如果表里没答案，就强制命令它去挖文本
        if not is_sufficient:
            guidance = "**CRITICAL**: The Table is KNOWN to lack the specific answer. You MUST extract the answer from the Textual Evidence."
        else:
            guidance = "**Note**: The Table likely contains the answer. Verify it against the Textual Evidence."

        # 获取基础子表数据
        subtable_df = self.df.loc[expanded_ids, col_info["selected_columns"]]
        # 文本检索与双向注入
        pruned_text_str = ""

        if self.text_embeddings is not None:
            top_text_ids = self._get_top_k_indices(query_emb, self.text_embeddings, top_k=20)
            candidate_texts = [self.raw_text_list[i] for i in top_text_ids]
            # 交给 pruning 函数做最后的内容精简 (取 Top 50%)
            pruned_units = self._retrieve_and_prune_text(query_emb, anchor_entities, candidate_texts)

            # 注入引用信息,利用上一步的向量做表文对齐
            injection_result = self._inject_cross_references(subtable_df, pruned_units)
            final_table_md = injection_result["table_md"]
            pruned_text_str = injection_result["text_str"]
        else:
            final_table_md = subtable_df.to_markdown(index=False)

        # 6. NLI 校验与显式打印
        # relevant_docs = [d['text'] for d in self.documents if d['row_id'] in expanded_ids]
        # nli_signals = self._verify_evidence(relevant_docs, pruned_text_str)
        # if nli_signals:
        #     print(f"\n🧠 [NLI Logic Check] Found {len(nli_signals)} signals:")
        #     for s in nli_signals:
        #         print(f"  - {s}")
        # else:
        #     print("\n🧠 [NLI Logic Check] No strong entailment or contradiction found.")

        return guidance, final_table_md, pruned_text_str

    # =========================================================================
    # 最终融合推理 (Hybrid Inference)
    # =========================================================================
    def query(self, question: str) -> str:
        """
        推理入口
        """
        guidance, final_table_md, pruned_text_str = self.retrieve_aligned_context(question)

        # 7. 生成
        final_prompt = f"""
    You are a factual reasoning assistant. Answer the question based on the evidence provided below.
    Rules:
1. **Check Table Sufficiency**: {guidance}

    ### 1. Structured Table Evidence (Key Rows & Columns)
    {final_table_md}
    ### 2. Supporting Textual Evidence (Extracted Facts)
    {pruned_text_str}
    - Question: {question}
    
PLEASE OUTPUT WITH THE FOLLOWING FORMAT:
<Answer>: direct answer
    """

        print("\n📝 [Final Prompt Context Preview]:")
        print(f"--- Table ---\n{final_table_md}\n--- Text ---\n{pruned_text_str}\n")

        # 4. 生成答案
        response = get_chat_result(
            messages=[{"role": "user", "content": final_prompt}],
            llm_config=self.llm_config
        )

        return response.content
