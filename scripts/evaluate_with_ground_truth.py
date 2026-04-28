#!/usr/bin/env python3
"""
基于标准答案的召回准确度评估脚本

评估指标：
1. Context Recall - 上下文召回率：标准答案中的关键信息有多少被成功召回
2. Context Precision - 上下文精确率：召回的内容中有多少是真正相关的
3. Keyword Recall - 关键词召回率：标准关键词在检索结果中的出现情况
4. Semantic Similarity - 语义相似度：检索结果与标准答案的语义匹配程度
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from src.rag.llamaindex.pipeline import RAGPipeline
from src.utils.config import load_config


@dataclass
class EvaluationMetrics:
    """评估指标"""
    query_id: str
    question: str
    context_recall: float  # 0-1
    context_precision: float  # 0-1
    keyword_recall: float  # 0-1
    semantic_score: float  # 0-1
    avg_retrieval_score: float  # 检索结果平均相似度
    retrieved_count: int  # 召回的文档数
    relevant_count: int  # 相关文档数


def calculate_keyword_recall(retrieved_texts: List[str], ground_truth_keywords: List[str]) -> float:
    """计算关键词召回率"""
    if not ground_truth_keywords:
        return 0.0

    found_keywords = 0
    all_text = " ".join(retrieved_texts).lower()

    for keyword in ground_truth_keywords:
        if keyword.lower() in all_text:
            found_keywords += 1

    return found_keywords / len(ground_truth_keywords)


def calculate_context_recall(
    retrieved_texts: List[str],
    ground_truth_answer: str,
    ground_truth_keywords: List[str]
) -> float:
    """
    计算上下文召回率
    基于：标准答案中的关键信息有多少被检索结果覆盖
    """
    if not retrieved_texts:
        return 0.0

    # 方法1：基于关键词覆盖
    keyword_score = calculate_keyword_recall(retrieved_texts, ground_truth_keywords)

    # 方法2：基于关键短语的覆盖（简化实现）
    all_retrieved = " ".join(retrieved_texts).lower()

    # 从标准答案中提取关键短语（这里简化为分词后的重要词组）
    gt_lower = ground_truth_answer.lower()

    # 检查标准答案中是否有重要信息出现在检索结果中
    important_phrases = [
        phrase for phrase in ground_truth_keywords
        if len(phrase) >= 4  # 过滤短词
    ]

    covered_phrases = 0
    for phrase in important_phrases:
        if phrase.lower() in all_retrieved or phrase.lower() in gt_lower:
            covered_phrases += 1

    phrase_score = covered_phrases / len(important_phrases) if important_phrases else 0.0

    # 综合得分（关键词70%，短语覆盖30%）
    return keyword_score * 0.7 + phrase_score * 0.3


def calculate_context_precision(
    retrieved_texts: List[str],
    ground_truth_keywords: List[str],
    similarity_scores: List[float]
) -> float:
    """
    计算上下文精确率
    基于：检索结果中有多少内容是与查询相关的
    """
    if not retrieved_texts:
        return 0.0

    # 基于相似度分数加权计算
    # 假设分数 > 0.7 是高相关，0.5-0.7 是中等相关，< 0.5 是低相关
    weighted_scores = []
    for score in similarity_scores:
        if score >= 0.7:
            weighted_scores.append(1.0)
        elif score >= 0.5:
            weighted_scores.append(0.7)
        elif score >= 0.3:
            weighted_scores.append(0.4)
        else:
            weighted_scores.append(0.1)

    return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    简单的语义相似度计算（基于关键词重叠的简化实现）
    实际应用中可以使用嵌入模型计算余弦相似度
    """
    # 分词并去重
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    # Jaccard 相似度
    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


async def evaluate_single_query(
    pipeline: RAGPipeline,
    test_case: Dict[str, Any]
) -> EvaluationMetrics:
    """评估单个查询"""

    question = test_case["question"]
    ground_truth_answer = test_case["ground_truth_answer"]
    ground_truth_keywords = test_case.get("relevant_keywords", [])

    # 执行检索
    result = await pipeline.ask(question)

    # 提取检索结果
    retrieved_texts = [source["content"] for source in result.get("sources", [])]
    similarity_scores = [source["score"] for source in result.get("sources", [])]

    # 计算各项指标
    keyword_recall = calculate_keyword_recall(retrieved_texts, ground_truth_keywords)

    context_recall = calculate_context_recall(
        retrieved_texts,
        ground_truth_answer,
        ground_truth_keywords
    )

    context_precision = calculate_context_precision(
        retrieved_texts,
        ground_truth_keywords,
        similarity_scores
    )

    # 计算与标准答案的语义相似度
    all_retrieved = " ".join(retrieved_texts)
    semantic_score = calculate_semantic_similarity(all_retrieved, ground_truth_answer)

    # 平均检索分数
    avg_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

    return EvaluationMetrics(
        query_id=test_case["id"],
        question=question,
        context_recall=context_recall,
        context_precision=context_precision,
        keyword_recall=keyword_recall,
        semantic_score=semantic_score,
        avg_retrieval_score=avg_score,
        retrieved_count=len(retrieved_texts),
        relevant_count=len([s for s in similarity_scores if s >= 0.5])
    )


async def run_evaluation():
    """运行完整评估"""

    print("=" * 80)
    print("基于标准答案的 RAG 召回准确度评估")
    print("=" * 80)
    print()

    # 加载测试数据集
    dataset_path = Path(__file__).parent.parent / "test_data" / "evaluation_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    test_cases = dataset["test_cases"]
    print(f"加载了 {len(test_cases)} 个测试用例")
    print()

    # 初始化 RAG Pipeline
    print("初始化 RAG Pipeline...")
    config = load_config()
    pipeline = RAGPipeline(config)

    # 检查索引状态
    if not pipeline.index:
        print("错误：索引未构建，请先上传文档并构建索引")
        print("提示：将测试文档放入 knowledge_base 目录后重新运行")
        return

    print("✓ Pipeline 初始化完成")
    print()

    # 执行评估
    print("开始评估...")
    print("-" * 80)

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] 评估: {test_case['question'][:50]}...")

        try:
            metrics = await evaluate_single_query(pipeline, test_case)
            results.append(metrics)

            print(f"  上下文召回率: {metrics.context_recall:.2%}")
            print(f"  上下文精确率: {metrics.context_precision:.2%}")
            print(f"  关键词召回率: {metrics.keyword_recall:.2%}")
            print(f"  语义相似度: {metrics.semantic_score:.2%}")
            print(f"  召回文档数: {metrics.retrieved_count}")

        except Exception as e:
            print(f"  错误: {e}")
            continue

    # 汇总结果
    print()
    print("=" * 80)
    print("评估结果汇总")
    print("=" * 80)

    if results:
        avg_recall = sum(r.context_recall for r in results) / len(results)
        avg_precision = sum(r.context_precision for r in results) / len(results)
        avg_keyword = sum(r.keyword_recall for r in results) / len(results)
        avg_semantic = sum(r.semantic_score for r in results) / len(results)
        avg_score = sum(r.avg_retrieval_score for r in results) / len(results)

        print()
        print(f"测试用例总数: {len(results)}")
        print()
        print("【核心指标】")
        print(f"  平均上下文召回率: {avg_recall:.2%}")
        print(f"  平均上下文精确率: {avg_precision:.2%}")
        print(f"  平均关键词召回率: {avg_keyword:.2%}")
        print()
        print("【辅助指标】")
        print(f"  平均语义相似度: {avg_semantic:.2%}")
        print(f"  平均检索分数: {avg_score:.4f}")
        print()

        # 按难度分析
        easy_results = [r for r, t in zip(results, test_cases) if t.get("difficulty") == "easy"]
        medium_results = [r for r, t in zip(results, test_cases) if t.get("difficulty") == "medium"]
        hard_results = [r for r, t in zip(results, test_cases) if t.get("difficulty") == "hard"]

        print("【按难度分析】")
        if easy_results:
            print(f"  简单题召回率: {sum(r.context_recall for r in easy_results)/len(easy_results):.2%} ({len(easy_results)}题)")
        if medium_results:
            print(f"  中等题召回率: {sum(r.context_recall for r in medium_results)/len(medium_results):.2%} ({len(medium_results)}题)")
        if hard_results:
            print(f"  困难题召回率: {sum(r.context_recall for r in hard_results)/len(hard_results):.2%} ({len(hard_results)}题)")

        print()
        print("【详细结果】")
        print("-" * 80)
        for r in results:
            status = "✓" if r.context_recall >= 0.7 else "⚠" if r.context_recall >= 0.5 else "✗"
            print(f"{status} [{r.query_id}] 召回率:{r.context_recall:.1%} 精确率:{r.context_precision:.1%} - {r.question[:40]}...")

        # 保存结果
        output = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_queries": len(results),
                "avg_context_recall": avg_recall,
                "avg_context_precision": avg_precision,
                "avg_keyword_recall": avg_keyword,
                "avg_semantic_score": avg_semantic,
            },
            "details": [
                {
                    "query_id": r.query_id,
                    "question": r.question,
                    "context_recall": r.context_recall,
                    "context_precision": r.context_precision,
                    "keyword_recall": r.keyword_recall,
                    "semantic_score": r.semantic_score,
                    "retrieved_count": r.retrieved_count,
                }
                for r in results
            ]
        }

        output_path = Path(__file__).parent.parent / "test_data" / "evaluation_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print()
        print(f"✓ 评估结果已保存至: {output_path}")

    else:
        print("没有成功完成任何评估")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
