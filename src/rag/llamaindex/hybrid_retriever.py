"""混合检索实现 - 结合向量检索和关键词检索"""

from typing import List
from collections import defaultdict

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, BM25Retriever
from llama_index.core.schema import NodeWithScore


def rrf_fusion(
    vector_results: List[NodeWithScore],
    bm25_results: List[NodeWithScore],
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
    k: int = 60
) -> List[NodeWithScore]:
    """RRF (Reciprocal Rank Fusion) 融合算法"""
    scores = defaultdict(float)
    node_map = {}

    for rank, node in enumerate(vector_results, start=1):
        node_id = node.node.node_id
        scores[node_id] += vector_weight / (k + rank)
        node_map[node_id] = node.node

    for rank, node in enumerate(bm25_results, start=1):
        node_id = node.node.node_id
        scores[node_id] += bm25_weight / (k + rank)
        if node_id not in node_map:
            node_map[node_id] = node.node

    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [NodeWithScore(node=node_map[nid], score=score) for nid, score in sorted_nodes]


class HybridRetriever:
    """混合检索器 - 融合向量检索和 BM25 检索结果"""

    def __init__(
        self,
        index: VectorStoreIndex,
        vector_retriever: VectorIndexRetriever,
        top_k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        self.index = index
        self.vector_retriever = vector_retriever
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.bm25_retriever = self._init_bm25_retriever()

    def _init_bm25_retriever(self) -> BM25Retriever:
        nodes = list(self.index.docstore.docs.values())
        return BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=self.top_k * 2
        )

    async def aretrieve(self, query: str) -> List[NodeWithScore]:
        vector_results = await self.vector_retriever.aretrieve(query)
        bm25_results = await self.bm25_retriever.aretrieve(query)

        fused_results = rrf_fusion(
            vector_results,
            bm25_results,
            self.vector_weight,
            self.bm25_weight
        )

        return fused_results[:self.top_k]
