#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG智能对话系统核心模块
实现基于检索增强生成的智能对话功能
"""

import os
import time
import logging
import json  # <--- 新增：用于解析JSON数据
from typing import List, Dict, Any, Optional
from pathlib import Path

# 环境变量加载
from dotenv import load_dotenv

load_dotenv()

# LangChain相关导入
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document  # <--- 新增：用于构建文档对象
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# HTTP请求
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepSeekLLM(LLM):
    """
    DeepSeek大语言模型接口封装
    实现LangChain LLM基类，支持API调用和重试机制
    """

    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_retries: int = 3

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat", temperature: float = 0.7, max_retries: int = 3):
        super().__init__(api_key=api_key, base_url=base_url, model=model,
                         temperature=temperature, max_retries=max_retries)

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """
        调用DeepSeek API生成回答
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 2000
        }

        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.warning(f"API调用失败，状态码: {response.status_code}, 尝试次数: {attempt + 1}")

            except Exception as e:
                logger.error(f"API调用异常: {str(e)}, 尝试次数: {attempt + 1}")

            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避

        raise Exception("API调用失败，已达到最大重试次数")


class RAGSystem:
    """
    RAG智能对话系统主类
    整合文档处理、向量存储、检索和生成功能
    """

    def __init__(self):
        """
        初始化RAG系统
        """
        # 从环境变量获取配置
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        self.base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 512))  # 医疗文本建议改小一点，这里设为512
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 64))
        self.max_retries = int(os.getenv('MAX_RETRIES', 3))

        if not self.api_key:
            raise ValueError("请在.env文件中设置DEEPSEEK_API_KEY")

        # 初始化组件
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.memory = None
        self.qa_chain = None

        # 初始化嵌入模型
        self._init_embeddings()

        # 初始化对话记忆
        self._init_memory()

        logger.info("RAG系统初始化完成")

    def _init_embeddings(self):
        """
        初始化嵌入模型
        修改：使用中文效果更好的 BAAI/bge-small-zh-v1.5
        """
        try:
            logger.info("尝试初始化本地HuggingFace嵌入模型 (BGE-Chinese)...")
            # 使用北京智源的 BGE 中文模型，适合中文医疗语境
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={
                    'device': 'cpu',  # 如果有GPU可改为 'cuda'
                    'trust_remote_code': False
                },
                cache_folder="./models",
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("本地中文嵌入模型(BGE)初始化成功")
        except Exception as e:
            logger.warning(f"本地嵌入模型初始化失败: {str(e)}")
            logger.info("切换到离线TF-IDF嵌入模式...")
            try:
                from offline_embeddings import TFIDFEmbeddings
                self.embeddings = TFIDFEmbeddings()
                logger.info("TF-IDF嵌入模型初始化成功")
            except ImportError:
                logger.warning("使用简单文本匹配模式（功能受限）")
                self.embeddings = None

    def _init_memory(self):
        """
        初始化对话记忆
        """
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        logger.info("对话记忆初始化成功")

    def load_medical_data(self, json_path: str) -> List[Document]:
        """
        新增：加载医疗JSON数据集

        Args:
            json_path: JSON文件路径

        Returns:
            List[Document]: 文档对象列表
        """
        if not os.path.exists(json_path):
            logger.warning(f"医疗数据集不存在: {json_path}")
            return []

        logger.info(f"正在加载医疗数据集: {json_path}")
        documents = []

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 处理数据，转换为 Document 对象
            # 假设数据格式为 [{"instruction": "...", "output": "..."}, ...]
            for i, item in enumerate(data):
                # 兼容不同的字段名
                q = item.get("instruction", "") or item.get("question", "")
                a = item.get("output", "") or item.get("answer", "")

                if q and a:
                    # 核心策略：将问题和答案拼在一起作为知识块
                    content = f"问题：{q}\n答案：{a}"

                    # 添加元数据，用于引用和溯源
                    metadata = {
                        "source": "医疗知识库",
                        "id": i,
                        "original_question": q
                    }

                    documents.append(Document(page_content=content, metadata=metadata))

            logger.info(f"医疗数据加载完成，共 {len(documents)} 条记录")
            return documents

        except Exception as e:
            logger.error(f"加载医疗数据失败: {str(e)}")
            return []

    def load_pdf_documents(self, pdf_paths: List[str]) -> List[Document]:
        """
        加载PDF文档并进行文本分割
        """
        all_documents = []

        # 使用递归字符分割器，更适合长文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？"]
        )

        for pdf_path in pdf_paths:
            try:
                if not os.path.exists(pdf_path):
                    logger.warning(f"文件不存在: {pdf_path}")
                    continue

                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                split_docs = text_splitter.split_documents(documents)
                all_documents.extend(split_docs)

                logger.info(f"成功加载文档: {pdf_path}, 分割为 {len(split_docs)} 个片段")

            except Exception as e:
                logger.error(f"加载文档失败 {pdf_path}: {str(e)}")

        logger.info(f"总共加载 {len(all_documents)} 个文档片段")
        return all_documents

    def build_vectorstore(self, documents: List[Document], save_path: str = None) -> bool:
        """
        构建向量数据库
        """
        try:
            if not documents:
                logger.warning("没有文档可以构建向量数据库")
                return False

            # 检查嵌入模型是否可用
            if self.embeddings is None:
                logger.warning("嵌入模型不可用，使用TF-IDF文档存储")
                from offline_embeddings import TFIDFEmbeddings
                self.tfidf_embeddings = TFIDFEmbeddings()
                self.documents = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
                self.tfidf_embeddings.embed_documents(self.documents)
                logger.info(f"TF-IDF文档存储构建成功，包含 {len(self.documents)} 个文档")
                return True

            # 构建FAISS向量数据库
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            # 保存向量数据库
            if save_path:
                try:
                    self.vectorstore.save_local(save_path)
                    logger.info(f"向量数据库已保存到: {save_path}")
                except Exception as save_error:
                    logger.warning(f"保存向量数据库失败: {str(save_error)}")

            logger.info("向量数据库构建成功")
            return True

        except Exception as e:
            logger.error(f"向量数据库构建失败: {str(e)}")
            # 备选方案：TF-IDF
            try:
                from offline_embeddings import TFIDFEmbeddings
                self.tfidf_embeddings = TFIDFEmbeddings()
                self.documents = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
                self.tfidf_embeddings.embed_documents(self.documents)
                logger.info(f"已切换到TF-IDF文档存储模式，包含 {len(self.documents)} 个文档")
                return True
            except Exception as fallback_error:
                logger.error(f"备选方案也失败: {str(fallback_error)}")
                return False

    def load_vectorstore(self, load_path: str) -> bool:
        """
        加载已保存的向量数据库
        """
        try:
            self.vectorstore = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"向量数据库加载成功: {load_path}")
            return True
        except Exception as e:
            logger.error(f"向量数据库加载失败: {str(e)}")
            return False

    def init_qa_chain(self, temperature: float = 0.7) -> bool:
        """
        初始化问答链
        """
        try:
            self.llm = DeepSeekLLM(
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=temperature,
                max_retries=self.max_retries
            )

            if self.vectorstore:
                logger.info("使用标准问答模式（向量检索）")
                # 创建检索器，k值可以适当调大以获取更多医疗背景
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )

                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True,
                    verbose=True,
                    output_key="answer"
                )
                logger.info("标准问答链初始化成功")
            else:
                logger.info("使用简单问答模式（无向量库）")
                self.qa_chain = "simple_mode"

            return True

        except Exception as e:
            logger.error(f"问答链初始化失败: {str(e)}")
            return False

    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        提问并获取回答（包含Prompt工程优化）
        """
        if not self.qa_chain:
            return {
                "answer": "系统未初始化，请先初始化问答链",
                "source_documents": [],
                "error": "System not initialized",
                "success": False
            }

        try:
            start_time = time.time()

            # 定义医疗领域的Prompt模板
            medical_prompt_template = """你是一名专业的医疗健康助手。请基于以下已知的医疗知识库内容来回答用户的问题。

已知医疗知识：
{context}

用户问题：{question}

回答要求：
1. 请根据上述“已知医疗知识”进行回答，不要编造事实。
2. 语言要专业、亲切、客观。
3. 如果知识库中没有相关信息，请明确告知“我的知识库中暂时没有关于此问题的记录，建议咨询专业医生”，不要随意瞎编。
4. 在回答结尾，如果确实引用了知识库，请标注“[基于知识库回答]”。

请开始回答："""

            # ------------------------------------------------------------------
            # 分支 1: 简单模式 (但可能持有 vectorstore 或 tfidf)
            # ------------------------------------------------------------------
            if self.qa_chain == "simple_mode":
                relevant_docs = []
                mode_name = "simple"

                if self.vectorstore:
                    # 有向量数据库，手动检索
                    try:
                        retriever = self.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 3}
                        )
                        relevant_docs = retriever.get_relevant_documents(question)
                    except Exception as e:
                        logger.warning(f"检索失败: {str(e)}")

                elif hasattr(self, 'tfidf_embeddings') and self.tfidf_embeddings is not None and hasattr(self,
                                                                                                         'documents') and self.documents:
                    # TF-IDF 模式
                    docs_info = self.tfidf_embeddings.similarity_search(question, self.documents, k=3)
                    # 将 TF-IDF 结果转换为伪 Document 对象以便统一处理
                    for info in docs_info:
                        relevant_docs.append(Document(
                            page_content=info['content'],
                            metadata={"score": info['score'], "index": info['index'], "source": "TF-IDF检索"}
                        ))

                # 构建上下文和Prompt
                if relevant_docs:
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    prompt = medical_prompt_template.format(context=context, question=question)
                else:
                    # 无任何文档时的保底回答
                    prompt = f"你是一名医疗助手。用户问：{question}。请回答，并提醒用户由于缺乏知识库支持，建议咨询医生。"

                # 调用LLM
                answer = self.llm._call(prompt)

                # 构建返回结果
                source_info = [{
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                } for doc in relevant_docs]

                return {
                    "answer": answer,
                    "source_documents": source_info,
                    "response_time": time.time() - start_time,
                    "success": True,
                    "mode": mode_name
                }

            # ------------------------------------------------------------------
            # 分支 2: 标准模式 (ConversationalRetrievalChain)
            # ------------------------------------------------------------------
            else:
                # 即使是 Chain 模式，我们也手动控制 Prompt 流程以保证 Prompt 效果
                # 因为默认的 Chain 内部 Prompt 比较难改，这里采用手动检索 + LLM 的方式覆盖

                # 1. 检索
                retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                relevant_docs = retriever.get_relevant_documents(question)

                # 2. 构建 Prompt
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                prompt = medical_prompt_template.format(context=context, question=question)

                # 3. 生成
                answer = self.llm._call(prompt)

                # 4. 格式化来源
                source_info = [{
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                } for doc in relevant_docs]

                return {
                    "answer": answer,
                    "source_documents": source_info,
                    "response_time": time.time() - start_time,
                    "success": True,
                    "mode": "standard"
                }

        except Exception as e:
            logger.error(f"问答处理失败: {str(e)}")
            return {
                "answer": f"抱歉，处理您的问题时出现错误: {str(e)}",
                "source_documents": [],
                "error": str(e),
                "success": False
            }

    def clear_memory(self):
        """
        清空对话记忆
        """
        if self.memory:
            self.memory.clear()
            logger.info("对话记忆已清空")

    def get_memory_summary(self) -> str:
        """
        获取对话记忆摘要
        """
        if self.memory and hasattr(self.memory, 'chat_memory'):
            messages = self.memory.chat_memory.messages
            if messages and len(messages) > 0:
                return f"当前对话历史包含 {len(messages)} 条消息"
        return "暂无对话历史"


if __name__ == "__main__":
    # 简单的本地测试逻辑
    print("正在测试 RAGSystem...")
    rag = RAGSystem()
    print("初始化完成。请在 main.py 中运行完整应用。")
    rag.init_qa_chain()
    print(rag.ask_question("糖尿病是什么"))