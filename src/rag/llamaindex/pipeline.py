"""LlamaIndex RAG Pipeline - 完整的检索增强生成流程"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.embeddings import MockEmbedding

# 设置默认 embed_model 避免 OpenAI 检查
Settings.embed_model = MockEmbedding(embed_dim=1024)
from llama_index.core.schema import TextNode, Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore as LlamaIndexChromaStore

from src.rag.components.factory import create_llm, create_embedder, create_reranker
from src.rag.components import *
from src.rag.components.llms.llama_index_adapter import LlamaIndexLLMAdapter
from src.rag.llamaindex.hybrid_retriever import HybridRetriever


class SmartTextProcessor:
    """智能文本处理器

    功能：
    1. 换行处理：单个换行替换为空格，保留双换行作为段落分隔符
    2. 文本清洗：合并多余空格，清理页眉页脚页码水印
    3. 结构还原：识别标题、列表、表格、代码块
    4. 智能分块：以段落为最小单位，支持不同文档类型的块大小
    """

    CHUNK_SIZES = {
        'general': (200, 600),
        'technical': (300, 800),
    }
    OVERLAP_RATIO = 0.125  # 50/400 = 12.5%
    TARGET_CHUNK_SIZE = 400
    OVERLAP_SIZE = 50  # 15% 重叠率

    # 文档类型检测关键字
    TECHNICAL_KEYWORDS = [
        '代码', '函数', 'api', '接口', '算法', '实现', '配置', '部署',
        '协议', '规范', '标准', '技术', '架构', '设计模式', 'class',
        'function', 'def ', 'import ', 'module', 'package'
    ]

    def __init__(self, doc_type: str = 'auto'):
        self.doc_type = doc_type
        if doc_type == 'auto':
            self.chunk_size = self.CHUNK_SIZES['general']
        else:
            self.chunk_size = self.CHUNK_SIZES.get(doc_type, self.CHUNK_SIZES['general'])

    def process(self, text: str, metadata: Dict = None) -> List[TextNode]:
        """处理文本，返回智能切分后的节点列表"""
        # 步骤1：基础清洗
        text = self._basic_clean(text)

        # 步骤2：检测文档类型
        if self.doc_type == 'auto':
            self._detect_doc_type(text)

        # 步骤3：结构解析
        blocks = self._parse_structure(text)

        # 步骤4：智能分块
        nodes = self._smart_chunk(blocks, metadata)

        return nodes

    def _basic_clean(self, text: str) -> str:
        """基础文本清洗"""
        if not text:
            return ''

        # 1. 移除控制字符（保留换行和回车）
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
        text = ''.join(c for c in text if not (0x00 <= ord(c) <= 0x1F and c not in '\n\r\t'))

        # 2. 单个换行替换为空格，保留双换行作为分段符
        # 先保护双换行
        text = text.replace('\r\n', '\n')  # 统一换行符
        text = text.replace('\r', '\n')

        # 处理3个以上换行，合并为双换行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 保护双换行，单换行替换为空格
        text = text.replace('\n\n', '\x00PARA\x00')
        text = text.replace('\n', ' ')
        text = text.replace('\x00PARA\x00', '\n\n')

        # 3. 合并连续空格
        text = re.sub(r' {2,}', ' ', text)

        # 4. 清理页眉页脚模式（数字+空格的页码）
        text = re.sub(r'\n\s*\d+\s*\n', '\n\n', text)

        # 5. 清理常见水印/页眉模式
        text = re.sub(r'(第\s*\d+\s*页|Page\s*\d+\s*of\s*\d+)', '', text, flags=re.IGNORECASE)

        return text.strip()

    def _detect_doc_type(self, text: str):
        """自动检测文档类型"""
        text_lower = text.lower()
        tech_score = sum(1 for kw in self.TECHNICAL_KEYWORDS if kw.lower() in text_lower)

        # 如果技术关键词密度高，判定为专业文档
        if tech_score > len(text) / 5000:  # 每5000字符出现一次技术关键词
            self.doc_type = 'technical'
            self.chunk_size = self.CHUNK_SIZES['technical']
        else:
            self.doc_type = 'general'
            self.chunk_size = self.CHUNK_SIZES['general']

    def _parse_structure(self, text: str) -> List[Dict]:
        """解析文档结构，识别标题、列表、表格、代码块"""
        blocks = []
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            block_type = self._detect_block_type(para)
            blocks.append({
                'type': block_type,
                'content': para,
                'is_structural': block_type in ['heading', 'code', 'table']
            })

        return blocks

    def _split_sentences(self, text: str) -> List[str]:
        """将段落文本按句子边界分割（句号优先）"""
        sentences = re.split(r'(?<=[。])', text)
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                result.append(s)
        return result

    def _detect_block_type(self, text: str) -> str:
        """检测块类型"""
        # 代码块检测
        if (text.startswith('    ') or  # 缩进代码
            text.startswith('```') or   # Markdown代码块
            re.match(r'^(function|class|def|import|const|let|var)\s', text) or
            re.match(r'^[\{\}\[\]\(\)<>]', text)):  # 括号开头
            return 'code'

        # 表格检测
        if '|' in text or re.match(r'^[\+\-]+[\+\-|\s]+[\+\-]+$', text):
            return 'table'

        # 标题检测
        if (re.match(r'^#{1,6}\s', text) or  # Markdown标题
            re.match(r'^\d+[\.、]\s*[^\n]{1,50}$', text) or  # 数字标题
            (len(text) < 100 and text and not text.endswith('。') and
             not text.endswith('，') and not text.endswith('；'))):
            return 'heading'

        # 列表检测
        if re.match(r'^[\*\-\+•·]\s', text) or re.match(r'^\d+[\.、\)]\s', text):
            return 'list'

        return 'paragraph'

    def _smart_chunk(self, blocks: List[Dict], metadata: Dict = None) -> List[TextNode]:
        """智能分块，保持结构完整性"""
        nodes = []
        current_chunk = []
        current_size = 0
        min_size, max_size = self.chunk_size

        for i, block in enumerate(blocks):
            block_text = block['content']
            block_len = len(block_text)
            block_type = block['type']

            if block['is_structural']:
                if current_chunk:
                    nodes.append(self._create_node(current_chunk, metadata))
                    current_chunk = []
                    current_size = 0
                nodes.append(self._create_node([block], metadata, is_structural=True))
                continue

            if block_len <= max_size:
                if current_size + block_len > max_size and current_size >= min_size:
                    nodes.append(self._create_node(current_chunk, metadata))
                    overlap_blocks = self._calculate_overlap(current_chunk)
                    current_chunk = overlap_blocks + [block]
                    current_size = sum(len(b['content']) for b in current_chunk)
                else:
                    current_chunk.append(block)
                    current_size += block_len
            else:
                if current_chunk:
                    nodes.append(self._create_node(current_chunk, metadata))
                    current_chunk = []
                    current_size = 0

                sentences = self._split_sentences(block_text)
                for sent in sentences:
                    sent_len = len(sent)
                    if current_size + sent_len > max_size and current_size >= min_size:
                        nodes.append(self._create_node(current_chunk, metadata))
                        overlap_blocks = self._calculate_overlap(current_chunk)
                        current_chunk = overlap_blocks
                        current_size = sum(len(b['content']) for b in current_chunk)
                    current_chunk.append({'type': block_type, 'content': sent, 'is_structural': False})
                    current_size += sent_len

        if current_chunk:
            nodes.append(self._create_node(current_chunk, metadata))

        return nodes

    def _calculate_overlap(self, blocks: List[Dict]) -> List[Dict]:
        """计算重叠内容（保留上一块的15%左右）"""
        if not blocks:
            return []

        total_len = sum(len(b['content']) for b in blocks)
        overlap_len = int(total_len * self.OVERLAP_RATIO)

        overlap_blocks = []
        current_overlap = 0

        # 从后向前取内容作为重叠
        for block in reversed(blocks):
            if current_overlap >= overlap_len:
                break
            overlap_blocks.insert(0, block)
            current_overlap += len(block['content'])

        return overlap_blocks

    def _create_node(self, blocks: List[Dict], metadata: Dict = None, is_structural: bool = False) -> TextNode:
        """创建文本节点"""
        content = '\n\n'.join(b['content'] for b in blocks)

        # ChromaDB 只支持简单元数据类型（str, int, float, None），不支持列表
        node_metadata = {
            'is_structural': is_structural,
            'block_types': ','.join(b['type'] for b in blocks),  # 列表转为逗号分隔字符串
            'chunk_size_type': self.doc_type,
        }
        if metadata:
            node_metadata.update(metadata)

        return TextNode(
            text=content,
            metadata=node_metadata
        )


class RAGPipeline:
    """RAG Pipeline 主类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.index: Optional[VectorStoreIndex] = None
        self._configure_settings()
        self._load_existing_index()

    def _load_existing_index(self):
        """尝试加载已有的 Chroma 索引"""
        import chromadb
        persist_dir = "./data/chroma_db"

        if not Path(persist_dir).exists():
            return

        try:
            chroma_client = chromadb.PersistentClient(path=persist_dir)
            # 检查集合是否存在
            collections = chroma_client.list_collections()
            if "knowledge_base" in [c.name for c in collections]:
                collection = chroma_client.get_collection("knowledge_base")
                vector_store = LlamaIndexChromaStore(
                    chroma_collection=collection
                )
                self.index = VectorStoreIndex.from_vector_store(vector_store)
                print(f"✅ 已加载现有索引: {persist_dir}")
        except Exception as e:
            print(f"⚠️ 加载现有索引失败: {e}")

    def _configure_settings(self):
        """配置 LlamaIndex 全局设置"""
        llm_config = self.config.get("llm", {})
        llm_provider = llm_config.get("provider", "qianfan")

        embedding_config = self.config.get("embedding", {})
        embed_provider = embedding_config.get("provider", "siliconflow")

        # 配置嵌入模型
        Settings.embed_model = create_embedder(embedding_config)

        # 配置 LLM
        llm = create_llm(llm_config)
        Settings.llm = LlamaIndexLLMAdapter(llm)

    def build_index(self, documents_path: str) -> VectorStoreIndex:
        """构建知识库索引（支持 Chroma 持久化，支持 PDF/Word/Excel/TXT）"""
        import chromadb

        # 配置文件加载器，支持 PDF 等格式
        from src.rag.components.loaders.pdf_loader import PDFLoader
        from src.rag.components.loaders.office_loader import DocxLoader, ExcelLoader

        file_extractor = {
            ".pdf": PDFLoader(),
            ".docx": DocxLoader(),
            ".xlsx": ExcelLoader(),
            ".xls": ExcelLoader(),
        }

        documents = SimpleDirectoryReader(
            documents_path,
            file_extractor=file_extractor
        ).load_data()

        # 使用智能文本处理器进行清洗和分块
        processor = SmartTextProcessor(doc_type='auto')
        all_nodes = []

        for doc in documents:
            # 提取文件类型信息用于元数据
            file_name = doc.metadata.get('file_name', '')
            file_type = doc.metadata.get('type', 'unknown')

            # 根据文件扩展名判断是否为技术文档
            is_technical = any(ext in file_name.lower() for ext in [
                '.py', '.js', '.java', '.cpp', '.go', '.rs',  # 代码文件
                '.md', '.rst', '.api', '.proto', '.yaml', '.yml', '.json'  # 配置/文档
            ])
            if is_technical:
                processor.doc_type = 'technical'
                processor.chunk_size = processor.CHUNK_SIZES['technical']
            else:
                processor.doc_type = 'general'
                processor.chunk_size = processor.CHUNK_SIZES['general']

            # 智能处理文档
            nodes = processor.process(doc.text, metadata={
                'file_name': file_name,
                'file_type': file_type,
                'source': doc.metadata.get('source', ''),
            })
            all_nodes.extend(nodes)

        # 使用 Chroma 持久化存储
        persist_dir = "./data/chroma_db"
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection("knowledge_base")
        vector_store = LlamaIndexChromaStore(
            chroma_collection=collection
        )

        # 创建 docstore 保存 nodes，供 HybridRetriever 使用
        docstore = SimpleDocumentStore()
        docstore.add_documents(all_nodes)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore
        )

        self.index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context
        )

        print(f"✅ 索引构建完成：共 {len(all_nodes)} 个节点，文档类型：{processor.doc_type}")
        return self.index

    def add_files_to_index(self, file_paths: list[str]) -> int:
        """增量添加文件到索引（只处理指定文件）"""
        import chromadb
        from src.rag.components.loaders.pdf_loader import PDFLoader
        from src.rag.components.loaders.office_loader import DocxLoader, ExcelLoader

        file_extractor = {
            ".pdf": PDFLoader(),
            ".docx": DocxLoader(),
            ".xlsx": ExcelLoader(),
            ".xls": ExcelLoader(),
        }

        all_nodes = []
        processor = SmartTextProcessor(doc_type='auto')

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                print(f"⚠️ 文件不存在: {file_path}")
                continue

            ext = path.suffix.lower()
            if ext in file_extractor:
                loader = file_extractor[ext]
                docs = loader.load_data(file_path)
            elif ext in ['.txt', '.md']:
                from llama_index.core import Document
                try:
                    text = path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    text = path.read_text(encoding='gbk', errors='ignore')
                docs = [Document(text=text, metadata={"file_name": path.name, "file_type": ext})]
            else:
                print(f"⚠️ 不支持的文件类型: {ext}")
                continue

            for doc in docs:
                file_name = doc.metadata.get('file_name', path.name)
                is_technical = any(ext in file_name.lower() for ext in [
                    '.py', '.js', '.java', '.cpp', '.go', '.rs',
                    '.md', '.rst', '.api', '.proto', '.yaml', '.yml', '.json'
                ])
                processor.doc_type = 'technical' if is_technical else 'general'
                processor.chunk_size = processor.CHUNK_SIZES.get(processor.doc_type, processor.CHUNK_SIZES['general'])

                nodes = processor.process(doc.text, metadata={
                    'file_name': file_name,
                    'file_type': ext,
                    'source': str(path),
                })
                all_nodes.extend(nodes)

        if not all_nodes:
            return 0

        if self.index is None:
            return self._create_index_with_nodes(all_nodes)

        self.index.insert_nodes(all_nodes)
        print(f"✅ 增量添加完成：新增 {len(all_nodes)} 个节点")
        return len(all_nodes)

    def _create_index_with_nodes(self, nodes: list) -> int:
        """用节点列表创建新索引"""
        import chromadb

        persist_dir = "./data/chroma_db"
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection("knowledge_base")
        vector_store = LlamaIndexChromaStore(chroma_collection=collection)

        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore
        )

        self.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context
        )
        print(f"✅ 新索引创建完成：共 {len(nodes)} 个节点")
        return len(nodes)

    async def ask(self, question: str, session_id: Optional[str] = None, selected_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """智能问答

        Args:
            question: 问题
            session_id: 会话ID
            selected_files: 选中的知识库文件列表，None表示全部，[]表示未选择
        """
        if not self.index:
            raise ValueError("索引未构建，请先调用 build_index()")

        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=20
        )
        hybrid_retriever = HybridRetriever(
            index=self.index,
            vector_retriever=vector_retriever,
            top_k=10
        )

        system_prompt = """你是一个直接的知识库检索助手。你只能根据检索到的内容回答问题，不允许添加任何检索结果中没有的信息。

严格规则：
1. 绝对禁止任何思考过程、推理过程、自我描述（如"好的"、"让我"、"我认为"等）
2. 只输出检索到的原文内容，可以修正语法和连贯性，但不能添加新含义
3. 如果检索结果中有多个相关内容，合并整理但不扩充
4. 如果检索结果不足以回答问题，直接说"检索结果不足以回答此问题"
5. 回答格式：直接给出答案，不要有任何开场白

回答示例：
- 错误："好的，我来回答这个问题。根据检索内容..."
- 错误："让我分析一下..."
- 正确："根据检索内容，答案是..."
- 正确：[直接给出检索到的内容]"""

        chat_engine = ContextChatEngine.from_defaults(
            retriever=hybrid_retriever,
            memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
            system_prompt=system_prompt
        )

        vector_results = await vector_retriever.aretrieve(question)

        import hashlib
        seen_hashes = set()
        sources = []

        for node in vector_results:
            file_name = node.node.metadata.get('file_name', '')

            if selected_files and file_name not in selected_files:
                continue

            content_preview = node.node.text[:200].strip()
            content_hash = hashlib.md5(content_preview.encode()).hexdigest()

            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            sources.append({
                "content": node.node.text,
                "score": float(node.score),
                "metadata": node.node.metadata
            })

            if len(sources) >= 10:
                break

        if sources:
            answer = "\n\n".join(s["content"] for s in sources[:5])
        else:
            answer = "未找到相关内容"

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id or "new_session"
        }

    def _filter_thinking_process(self, text: str) -> str:
        """过滤AI的思考过程，只保留最终答案"""
        if not text:
            return text

        import re

        # 将文本按句子分割处理
        sentences = re.split(r'([。！？\n])', text)
        result_sentences = []
        
        # 思考过程模式 - 以第一人称或思考动词开头的句子
        thinking_starts = [
            r'^(好的|好|现在|让我|此外|整个|这部分|我得|我要|希望|嗯|啊|哦|那个|首先|其次|然后|最后|接下来|为了|因此|所以|但是|不过|然而)[,，：:]',
            r'^我[^。]*(思考|考虑)[^。]*',  # 以\"我\"开头且包含\"思考\"或\"考虑\"的句子
            r'^根据[^。]*?(思考|考虑)[^。]*',  # 只过滤包含\"思考\"或\"考虑\"的根据句
        ]
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            # 如果是标点符号，添加到前一个句子
            if sentence in ['。', '！', '？', '\n']:
                if result_sentences:
                    result_sentences[-1] += sentence
                i += 1
                continue
            
            # 检查是否是思考过程
            is_thinking = False
            for pattern in thinking_starts:
                if re.search(pattern, sentence.strip(), re.MULTILINE):
                    is_thinking = True
                    break
            
            if not is_thinking:
                result_sentences.append(sentence)
            
            i += 1

        text = ''.join(result_sentences)
        
        # 清理连续换行和开头结尾的空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    async def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """纯检索（不调用 LLM）"""
        if not self.index:
            raise ValueError("索引未构建")

        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k
        )

        nodes = await vector_retriever.aretrieve(query)

        return [
            {
                "content": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata
            }
            for node in nodes
        ]
