import numpy as np
import pandas as pd
import json
import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import math
import re
from difflib import get_close_matches
from chat_utils import get_chat_result
from config import config_mapping
from utils.tool_utils import Embedder
import time
from contextlib import contextmanager
from collections import Counter
import math

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
        # self.nli_model_name = "models/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        # self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        # self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name).to(self.device)
        # self.nli_model.eval()  # 务必开启 eval 模式，关闭 Dropout
        # self.nli_labels = ["entailment", "neutral", "contradiction"]

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

    def _smart_format(self, template: str, row_dict: Dict) -> str:
        """
        填充器：允许 LLM 稍微写错列名，代码负责自动纠正。
        """
        # 1. 找出模板里所有需要的 {Key}
        # 比如模板是 "{Browser} uses {Engine}." -> 提取出 ['Browser', 'Engine']
        needed_keys = re.findall(r'\{(.+?)\}', template)

        # 2. 准备实际的数据池
        actual_keys = list(row_dict.keys())
        # 创建一个归一化映射 (全小写 -> 真实Key)
        lower_map = {k.lower().strip(): k for k in actual_keys}

        # 3. 构建最终的填充字典
        final_mapping = {}

        for placeholder in needed_keys:
            # Case A: 完全匹配 (最完美)
            if placeholder in row_dict:
                final_mapping[placeholder] = row_dict[placeholder]
                continue

            # Case B: 忽略大小写和空格匹配
            clean_placeholder = placeholder.lower().strip()
            if clean_placeholder in lower_map:
                real_key = lower_map[clean_placeholder]
                final_mapping[placeholder] = row_dict[real_key]
                continue

            # Case C: 模糊匹配 (difflib)
            # 比如 LLM 写了 {Layout}，实际是 {Current layout engine}
            # cutoff=0.6 表示只要有 60% 像就可以
            matches = get_close_matches(placeholder, actual_keys, n=1, cutoff=0.6)
            if matches:
                real_key = matches[0]
                final_mapping[placeholder] = row_dict[real_key]
                # print(f"🔧 Auto-fixed: {{{placeholder}}} -> '{real_key}'") # 调试用
            else:
                # Case D: 实在找不到，填个默认值，保证不崩
                final_mapping[placeholder] = "Unknown"

        # 4. 安全填充
        return template.format(**final_mapping)

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
                # 使用智能模糊填充，而不是死板的 format
                text = self._smart_format(py_template, row_dict)
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
        self.table_embeddings = F.normalize(raw_emb, p=2, dim=1).cpu()

        # 对外部文本列表进行向量化
        if self.raw_text_list and len(self.raw_text_list) > 0:
            print(f"⚡ Encoding {len(self.raw_text_list)} External Text Blocks...")
            self.text_embeddings = F.normalize(torch.tensor(self.embedder.encode(self.raw_text_list)), p=2, dim=1).cpu()
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
  "answer_in_table": true/false
}}
    """
        formatted_prompt = prompt.format(question=question, columns=', '.join(all_cols))
        response = get_chat_result(
            messages=[{"role": "user", "content": formatted_prompt}],
            tools=None,
            llm_config=self.llm_config
        )

        result = self._clean_json_response(response.content)
        # 1. 获取 LLM 想要保留的列
        selected = result.get("selected_columns", [])
        # 2. [关键修复] 强制注入 Primary Key (self.pk_col)
        # 无论 LLM 觉得需不需要，程序逻辑需要它
        if self.pk_col not in selected:
            # print(f"🔧 Auto-injecting PK column: {self.pk_col}")
            selected.insert(0, self.pk_col)

        # 3. 校验选出的列是否真的在表中
        final_selected = [c for c in selected if c in all_cols]
        if not final_selected:
            final_selected = all_cols

        print(f"🏷️ answer_in_table: {result['answer_in_table']}")
        result["selected_columns"] = final_selected
        return result

    def _analyze_query_intent(self, question: str) -> str:
        """
        分析问题意图：是简单的查值，还是复杂的聚合/排序
        """
        q_lower = question.lower()

        # 1. 聚合类关键词 (Aggregation)
        agg_keywords = ["how many", "sum", "average", "total", "percentage", "count", "amount"]
        if any(w in q_lower for w in agg_keywords):
            return "aggregation"

        # 2. 排序/比较类关键词 (Ranking)
        # 注意：包含 'second', 'most', 'top' 等
        rank_keywords = ["most", "least", "best", "worst", "top", "first", "second",
                         "third", "last", "rank", "sort", "highest", "lowest", "compare"]
        if any(w in q_lower for w in rank_keywords):
            return "ranking"

        # 3. 默认查值 (Retrieval)
        return "retrieval"

    def _expand_context_radius(self, anchor_ids: List[int], intent: str) -> List[int]:
        """
        根据意图自适应分配上下文行。
        intent: "retrieval" | "ranking" | "aggregation"
        """
        final_ids = set(anchor_ids)

        # === 场景 A: 简单查值 (Retrieval) ===
        # 策略：关注局部上下文
        # 逻辑：加上前后邻居，帮助理解上下文衔接
        if intent == "retrieval":
            for rid in anchor_ids:
                if rid > 0: final_ids.add(rid - 1)
                if rid < len(self.df) - 1: final_ids.add(rid + 1)

        # === 场景 B: 排名或聚合 (Ranking / Aggregation) ===
        else:
            # 1. 强制加入 Top-10 行
            top_n_count = 10
            for i in range(min(top_n_count, len(self.df))):
                final_ids.add(i)

        # === 最终处理 ===
        sorted_ids = sorted(list(final_ids))
        # 动态截断：如果是 Ranking 问题，尽量多给几行，防止榜单断裂
        limit = 25 if intent in ["ranking", "aggregation"] else 15

        return sorted_ids[:limit]

    def _retrieve_and_prune_text(self, query_emb: torch.Tensor, anchor_entities: List[str],
                                 retrieved_texts: List[str]) -> List[Dict]:
        """
        [Text Pruning] 双路召回版 (Dual-Route Retrieval)
        为了防止加权策略导致的“逆反”，我们采用分路录取策略：
        1. 语义通道：录取向量相似度最高的文本。
        2. 词汇通道：录取实体关键词匹配度最高的文本。
        最后取并集。
        """
        if not retrieved_texts: return []

        # --- 1. 准备 IDF 权重 (用于词汇通道) ---
        # 统计 anchor 实体的词频，计算简易 IDF
        all_anchor_tokens = []
        for ent in anchor_entities:
            tokens = [w.lower() for w in re.split(r'\W+', str(ent)) if len(w) > 2]
            all_anchor_tokens.extend(tokens)

        token_counts = Counter(all_anchor_tokens)
        num_anchors = max(1, len(anchor_entities))

        idf_weights = {}
        for token, count in token_counts.items():
            # 越稀有的实体词，权重越大
            idf_weights[token] = math.log(num_anchors / (count + 1)) + 1.0

        # --- 2. 文本预处理与去重 ---
        seen_units = set()
        for text in retrieved_texts:
            is_kv = len(re.findall(r'[:：|]', text)) > len(text) / 50
            units = re.split(r'[\n;]', text) if is_kv else re.split(r'(?<=[。？！?.])\s+', text)
            for u in units:
                u_clean = u.strip()
                if len(u_clean) > 5 and u_clean not in seen_units:
                    seen_units.add(u_clean)

        unique_units = list(seen_units)
        if not unique_units: return []

        # --- 3. 向量化 (语义通道) ---
        raw_embs = torch.tensor(self.embedder.encode(unique_units))
        unit_embs = torch.nn.functional.normalize(raw_embs, p=2, dim=1)

        if query_emb.dim() == 1:
            dense_scores = torch.matmul(unit_embs, query_emb).cpu().numpy()
        else:
            dense_scores = torch.matmul(unit_embs, query_emb.t()).squeeze().cpu().numpy()

        # --- 4. 评分计算 ---
        scored_units = []
        for i, text_unit in enumerate(unique_units):
            d_score = dense_scores[i]  # 语义分
            text_lower = text_unit.lower()

            # 计算词汇分 (Lexical Score)
            s_score = 0.0
            for token, weight in idf_weights.items():
                if token in text_lower:
                    s_score += weight

            scored_units.append({
                "text": text_unit,
                "embedding": unit_embs[i],
                "dense_score": d_score,
                "sparse_score": s_score,
                "original_index": i
            })

        # --- 5. 双路录取 (Dual Selection) ---

        # 预算分配：总共留 35~50 个
        # 语义通道占 70%，词汇通道占 30% (保证语义是主流，关键词是补充)
        total_budget = min(50, math.ceil(len(scored_units) * 0.6))
        total_budget = max(25, total_budget)  # 至少留 25 个

        semantic_budget = int(total_budget * 0.7)
        lexical_budget = total_budget - semantic_budget

        final_indices = set()

        # Route A: 语义优先 (Vector High Score)
        scored_units.sort(key=lambda x: x["dense_score"], reverse=True)
        for i in range(min(len(scored_units), semantic_budget)):
            final_indices.add(scored_units[i]["original_index"])

        # Route B: 词汇优先 (Keyword High Score)
        # 重新排序，这次看 sparse_score
        scored_units.sort(key=lambda x: x["sparse_score"], reverse=True)

        # 录取那些还没有被语义通道选中的“漏网之鱼”
        added_count = 0
        for unit in scored_units:
            if added_count >= lexical_budget:
                break
            if unit["original_index"] not in final_indices:
                # 只有当它确实包含关键词 (sparse_score > 0) 时才救回
                if unit["sparse_score"] > 0:
                    final_indices.add(unit["original_index"])
                    added_count += 1

        # --- 6. 组装最终结果 ---
        # 按照原始的语义分数排序输出，保证后续处理顺序正常
        final_result = []
        for i in range(len(unique_units)):
            if i in final_indices:
                # 找到对应的分数对象
                # 为了后续兼容，我们把 score 设为 dense_score，因为对齐阶段会重新算
                unit_obj = next(u for u in scored_units if u["original_index"] == i)
                final_result.append({
                    "text": unit_obj["text"],
                    "score": unit_obj["dense_score"],  # 保持 API 兼容
                    "embedding": unit_obj["embedding"]
                })

        # 再次按分数排序返回
        final_result.sort(key=lambda x: x["score"], reverse=True)
        return final_result

    # def _retrieve_and_prune_text(self, query_emb: torch.Tensor, anchor_entities: List[str],
    #                              retrieved_texts: List[str]) -> List[Dict]:
    #     """
    #     2. 自动判定 KV 结构与句子结构
    #     3. 基于 BGE 相似度与实体锚定打分
    #     """
    #     if not retrieved_texts: return []
    #
    #     entity_keywords = set()
    #     for ent in anchor_entities:
    #         for word in re.split(r'\W+', ent):  # 按非字母字符拆分
    #             if len(word) > 3:  entity_keywords.add(word.lower())
    #
    #     seen_units = set()  # 用于去重
    #     for text in retrieved_texts:
    #         # 自动判定 KV vs 纯文本结构
    #         is_kv = len(re.findall(r'[:：|]', text)) > len(text) / 50
    #         units = re.split(r'[\n;]', text) if is_kv else re.split(r'(?<=[。？！?.])\s+', text)
    #         for u in units:
    #             u_clean = u.strip()
    #             if len(u_clean) > 5 and u_clean not in seen_units:
    #                 seen_units.add(u_clean)
    #
    #     unique_units = list(seen_units)
    #     if not seen_units: return []
    #
    #     # 向量化 (增加手动归一化，确保后续计算准确)
    #     # raw_embs: [N, Dim]
    #     raw_embs = torch.tensor(self.embedder.encode(unique_units))
    #     unit_embs = torch.nn.functional.normalize(raw_embs, p=2, dim=1)
    #     # 打分 (Query vs Units)
    #     if query_emb.dim() == 1:
    #         scores = torch.matmul(unit_embs, query_emb)
    #     else:
    #         scores = torch.matmul(unit_embs, query_emb.t()).squeeze()
    #     scores = scores.cpu().numpy()
    #
    #     all_units = []
    #     for i, score in enumerate(scores):
    #         text_unit = unique_units[i]
    #         # 关键词加分
    #         if any(kw in text_unit.lower() for kw in entity_keywords):
    #             score += 0.2
    #         all_units.append({
    #             "text": text_unit,
    #             "score": score,
    #             "embedding": unit_embs[i]  # 带出向量，供下一步对齐使用
    #         })
    #
    #     # 保留前 50%
    #     all_units.sort(key=lambda x: x["score"], reverse=True)
    #     keep_count = min(80, math.ceil(len(all_units) * 0.5))  # 稍微放宽一点上限到，保证上下文
    #
    #     return all_units[:keep_count]

    def _inject_cross_references(self, sub_df: pd.DataFrame, pruned_units: List[Dict]) -> Dict[str, str]:
        """
        核心功能：通用混合检索对齐 (Robust Hybrid Alignment)
        不再使用硬阈值保送，而是使用加权融合。引入 IDF 思想,关键词匹配不能“命中一个就给满分”。命中稀有词（如 "Android"）给高分，命中普通词给低分。
        """
        if not pruned_units:
            return {"table_md": sub_df.to_markdown(index=False), "text_str": ""}

        # 1. [新增] 动态计算表格内的 IDF (词的稀缺度)
        all_tokens_flat = []
        for val in sub_df[self.pk_col]:
            # 简单分词，过滤短词
            tokens = [w.lower() for w in re.split(r'\W+', str(val)) if len(w) > 2]
            all_tokens_flat.extend(tokens)

        token_counts = Counter(all_tokens_flat)
        total_rows = len(sub_df)

        # 计算每个词的 IDF 权重: log(总行数 / (词频 + 1)) + 1
        # 稀有词权重高，高频词(如 Browser)权重低
        idf_weights = {}
        for token, count in token_counts.items():
            idf_weights[token] = math.log(total_rows / (count + 1)) + 1.0

        # 2. 准备向量
        # 确保都在 CPU 上计算
        unit_embs = torch.stack([u['embedding'] for u in pruned_units]).cpu()
        row_indices = sub_df.index.tolist()
        row_embs = self.table_embeddings[row_indices].cpu()

        # [K, M] 向量相似度矩阵
        dense_scores = torch.matmul(row_embs, unit_embs.t()).numpy()

        # 3. 容器
        row_refs = {i: [] for i in range(len(sub_df))}
        unit_labels = {j: set() for j in range(len(pruned_units))}

        # 4. 混合检索循环
        for r_idx in range(len(sub_df)):
            row_entity = str(sub_df.iloc[r_idx][self.pk_col])
            # 提取当前行的实体 tokens
            row_tokens = [w.lower() for w in re.split(r'\W+', row_entity) if len(w) > 2]

            candidates = []

            for u_idx in range(len(pruned_units)):
                # A. 稠密分 (Dense Score): 范围通常 -1 ~ 1
                d_score = dense_scores[r_idx][u_idx]

                # B. 稀疏分 (Sparse Score): 基于 IDF 加权
                text_content = pruned_units[u_idx]['text'].lower()

                s_score = 0.0
                for token in row_tokens:
                    if token in text_content:
                        # 命中稀有词加分多，命中高频词加分少
                        s_score += idf_weights.get(token, 1.0)

                # 归一化 Sparse Score (防止长实体分数无限膨胀)
                # 假设匹配了 2-3 个核心词就算很高了，封顶 1.0
                s_score = min(s_score / 4.0, 1.0)

                # C. 融合分 (Hybrid Score)
                # 0.7 * 向量 + 0.3 * 关键词
                final_score = 0.7 * d_score + 0.3 * s_score

                # [核心修复] 这里必须 Append 3个值，对应后面解包的 3个变量
                candidates.append((final_score, u_idx, d_score))

            # 5. 排序与截断
            # 按 final_score 降序排列
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_k = candidates[:5]

            # 6. 最终安全网 (Soft Threshold)
            # 这里解包 3 个值就不会报错了
            for f_score, u_idx, raw_score in top_k:
                # 只要混合分 > 0.45 就可以入选
                # 或者：虽然混合分略低，但原始向量分极高 (>0.65) 也可以入选
                if f_score > 0.45 or raw_score > 0.65:
                    # 记录时展示混合分数，方便调试
                    row_refs[r_idx].append(f"[{u_idx}]({f_score:.2f})")
                    unit_labels[u_idx].add(row_entity)

        # 7. 生成增强版表格
        view_df = sub_df.copy()
        view_df["Related Context IDs"] = [", ".join(refs) for refs in row_refs.values()]
        table_md = view_df.to_markdown(index=False)

        # 8. 生成增强版文本串
        formatted_texts = []
        for i, unit in enumerate(pruned_units):
            labels = sorted(list(unit_labels[i]))
            label_str = f"[Rel: {', '.join(labels)}]" if labels else ""
            formatted_texts.append(f"[{i}] {label_str} {unit['text']}")

        return {
            "table_md": table_md,
            "text_str": "\n".join(formatted_texts)
        }

    # def _verify_evidence(self, sub_table_facts: List[str], text_evidence: str) -> List[str]:
    #     """
    #     利用 Tokenizer 的 Batch 处理能力，一次性校验所有表格事实
    #     """
    #     if not text_evidence or not sub_table_facts:
    #         return []
    #
    #     verification_signals = []
    #     # 将文本证据作为统一的前提 (Premise)
    #     premise = text_evidence[:1500]
    #
    #     try:
    #         entail_idx = self.nli_labels.index("entailment")
    #         contra_idx = self.nli_labels.index("contradiction")
    #     except ValueError:
    #         # 兜底逻辑：如果 labels 设置不对，默认使用官方标准 0, 2
    #         entail_idx, contra_idx = 0, 2
    #
    #     # 1. 构造 Batch 输入对：[[Premise, Hypo1], [Premise, Hypo2], ...]
    #     pairs = [[premise, fact] for fact in sub_table_facts]
    #
    #     # 2. 调用 Tokenizer 的批处理功能
    #     # padding=True 会自动对齐长度，return_tensors="pt" 返回 PyTorch 张量
    #     inputs = self.nli_tokenizer(
    #         pairs,
    #         padding=True,
    #         truncation=True,
    #         max_length=512,
    #         return_tensors="pt"
    #     ).to(self.device)
    #
    #     # 3. 开启无梯度推理模式
    #     with torch.no_grad():
    #         outputs = self.nli_model(**inputs)
    #         # 对 logits 在最后一个维度（标签维度）做 Softmax，得到概率分布 [Batch_size, 3]
    #         predictions = torch.softmax(outputs.logits, dim=-1)
    #
    #     # 4. 解析结果 (对应官方标签顺序: entailment, neutral, contradiction)
    #     # 将结果转回 CPU 列表处理
    #     predictions = predictions.cpu().numpy()
    #
    #     for i, probs in enumerate(predictions):
    #         fact = sub_table_facts[i]
    #         entail_prob = probs[entail_idx]
    #         contra_prob = probs[contra_idx]
    #
    #         # 阈值判定：只有置信度够高才输出信号，减少噪声
    #         if entail_prob > 0.7:
    #             verification_signals.append(f"✅ Fact Verified: {fact[:60]}... (Conf: {entail_prob:.1%})")
    #         elif contra_prob > 0.7:
    #             verification_signals.append(f"❌ Conflict Detected: {fact[:60]}... (Conf: {contra_prob:.1%})")
    #
    #     return verification_signals

    def retrieve_aligned_context(self, question: str):
        """
        推理入口：结合自适应子表与精简 KV 文本
        """
        print(f"\n=== 🚀 Hybrid Query: {question} ===")
        query_emb_numpy = self.embedder.encode(question)
        query_emb = torch.tensor(query_emb_numpy).squeeze().cpu()

        # 1. 意图分析与锚点检索
        intent = self._analyze_query_intent(question)
        anchor_ids = self._get_top_k_indices(query_emb, self.table_embeddings, top_k=10)
        anchor_entities = [self.df.iloc[rid][self.pk_col] for rid in anchor_ids]
        expanded_ids = self._expand_context_radius(anchor_ids, intent)

        # 关键词检测：如果问题包含排序、最值、计数等词汇
        ranking_keywords = ["most", "least", "best", "worst", "top", "first", "second", "third", "last", "rank", "sort",
                            "highest", "lowest"]
        is_ranking_query = any(kw in question.lower() for kw in ranking_keywords)

        if is_ranking_query or intent == "ranking" or intent == "aggregation":
            print(f"📊 Detected Ranking/Aggregation Query: Preserving Table Structure...")
            # 策略 A：如果是小表 (50行以内)，干脆全给，不要让 LLM 猜
            if len(self.df) <= 50:
                expanded_ids = list(range(len(self.df)))
            # 策略 B：如果是大表，强制钉死前 10 行 (Pin Top-N)
            # 这样 LLM 就能看到 Rank 1, 2, 3... 从而建立正确的坐标系
            else:
                top_rows_count = 10
                # 确保不超过表长度
                top_ids = [i for i in range(min(top_rows_count, len(self.df)))]
                # 合并 语义检索行 + 头部行
                expanded_ids = sorted(list(set(expanded_ids + top_ids)))
        expanded_ids.sort()

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
            top_text_ids = self._get_top_k_indices(query_emb, self.text_embeddings, top_k=30)
            candidate_texts = [self.raw_text_list[i] for i in top_text_ids]
            # 交给 pruning 函数做最后的内容精简
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
    
Please format your output EXACTLY as follows:
{{
<Answer>: [The direct answer]
}}
    """

        print("\n📝 [Final Prompt Context Preview]:")
        print(f"--- Table ---\n{final_table_md}\n--- Text ---\n{pruned_text_str}\n")

        # 4. 生成答案
        response = get_chat_result(
            messages=[{"role": "user", "content": final_prompt}],
            llm_config=self.llm_config
        )

        return response.content
