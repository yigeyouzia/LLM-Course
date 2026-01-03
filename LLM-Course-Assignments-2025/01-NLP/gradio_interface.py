#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG智能对话系统Gradio界面模块
提供用户友好的Web交互界面
"""

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr
import tempfile
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from rag_system import RAGSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGInterface:
    """
    RAG系统Gradio界面封装类
    """

    def __init__(self):
        """
        初始化界面
        """
        self.rag_system = None
        self.uploaded_files = []
        self.chat_history = []

        # 创建临时目录存储上传的文件
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"临时目录创建: {self.temp_dir}")

    def upload_pdf(self, files) -> str:
        """
        处理PDF文件上传（支持增量添加到知识库）

        Args:
            files: 上传的文件列表

        Returns:
            str: 处理结果信息
        """
        if not files:
            return "请选择PDF文件上传"

        try:
            # 如果是第一次上传，初始化文件列表
            if not hasattr(self, 'uploaded_files') or self.uploaded_files is None:
                self.uploaded_files = []

            new_files = []
            valid_files = []
            skipped_files = []

            for file in files:
                if file is None:
                    continue

                # 检查文件扩展名
                if not file.name.lower().endswith('.pdf'):
                    logger.warning(f"跳过非PDF文件: {file.name}")
                    skipped_files.append(os.path.basename(file.name))
                    continue

                # 检查是否已经上传过该文件
                file_basename = os.path.basename(file.name)
                temp_path = os.path.join(self.temp_dir, file_basename)

                if temp_path in self.uploaded_files:
                    logger.info(f"文件已存在于知识库中: {file_basename}")
                    skipped_files.append(f"{file_basename} (已存在)")
                    continue

                # 复制文件到临时目录
                shutil.copy2(file.name, temp_path)

                self.uploaded_files.append(temp_path)
                new_files.append(temp_path)
                valid_files.append(file_basename)

            if not new_files:
                if skipped_files:
                    return f"没有新文件被添加。跳过的文件:\n" + "\n".join([f"• {name}" for name in skipped_files])
                else:
                    return "没有有效的PDF文件被上传"

            # 初始化RAG系统（如果还没有初始化）
            if self.rag_system is None:
                self.rag_system = RAGSystem()

            # 加载新文档
            new_documents = self.rag_system.load_pdf_documents(new_files)

            # 如果已有知识库，需要合并文档
            if hasattr(self, 'all_documents') and self.all_documents:
                self.all_documents.extend(new_documents)
                logger.info(f"向现有知识库添加 {len(new_documents)} 个新文档片段")
            else:
                self.all_documents = new_documents
                logger.info(f"创建新知识库，包含 {len(new_documents)} 个文档片段")

            documents = self.all_documents

            if not documents:
                return "PDF文件加载失败，请检查文件格式"

            # 重新构建向量数据库（包含所有文档）
            success = self.rag_system.build_vectorstore(documents)

            if not success:
                return "向量数据库构建失败"

            # 初始化QA链
            success = self.rag_system.init_qa_chain()

            if not success:
                return "问答系统初始化失败"

            # 构建状态消息
            total_files = len(self.uploaded_files)
            total_docs = len(documents)
            new_file_count = len(valid_files)

            status_msg = f"✅ 知识库更新成功！\n\n"
            status_msg += f"📊 知识库状态：\n"
            status_msg += f"• 总文件数：{total_files} 个PDF文件\n"
            status_msg += f"• 总文档片段：{total_docs} 个\n\n"

            if new_file_count > 0:
                status_msg += f"📁 本次新增文件 ({new_file_count} 个)：\n"
                status_msg += "\n".join([f"• {name}" for name in valid_files])

            if skipped_files:
                status_msg += f"\n\n⚠️ 跳过的文件：\n"
                status_msg += "\n".join([f"• {name}" for name in skipped_files])

            return status_msg

        except Exception as e:
            error_msg = f"文件处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg