import chromadb
import uuid
import asyncio
from typing import List, Dict, Tuple, Optional

from .base import (
    VectorDBBase,
    Document,
    ProcessingBatch,
    DEFAULT_BATCH_SIZE,
    MAX_RETRIES,
)
from astrbot.api import logger
from ..utils.embedding import EmbeddingSolutionHelper


class ChromaStore(VectorDBBase):
    def __init__(
        self, embedding_util: EmbeddingSolutionHelper, data_path: str, **kwargs
    ):
        super().__init__(embedding_util, data_path)
        self.client: Optional[chromadb.Client] = None
        self.doc_mappings: Dict[str, Dict[str, Document]] = {}
        # 遵循 faiss_store.py 的做法，使用 DEFAULT_BATCH_SIZE
        self.embedding_batch_size = DEFAULT_BATCH_SIZE

    async def initialize(self):
        logger.info(f"初始化 ChromaDB 客户端，路径: {self.data_path}...")
        try:
            # ChromaDB 客户端的IO操作是同步的，使用 to_thread 包装以避免阻塞
            self.client = await asyncio.to_thread(
                chromadb.PersistentClient, path=self.data_path
            )
            logger.info("ChromaDB 客户端初始化成功。")
        except Exception as e:
            logger.error(f"初始化 ChromaDB 客户端失败: {e}", exc_info=True)
            self.client = None
            raise

    async def create_collection(self, collection_name: str):
        if not self.client:
            raise ConnectionError("ChromaDB 客户端未初始化。")
        try:
            # 同步操作，使用 to_thread
            await asyncio.to_thread(self.client.create_collection, name=collection_name)
            self.doc_mappings[collection_name] = {}
            logger.info(f"ChromaDB 集合 '{collection_name}' 创建成功。")
        except chromadb.errors.DuplicateCollectionError:
            logger.info(f"ChromaDB 集合 '{collection_name}' 已存在。")
        except Exception as e:
            logger.error(
                f"创建 ChromaDB 集合 '{collection_name}' 时出错: {e}",
                exc_info=True,
            )
            raise

    async def collection_exists(self, collection_name: str) -> bool:
        if not self.client:
            return False
        try:
            # 同步操作，使用 to_thread
            collections = await asyncio.to_thread(self.client.list_collections)
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            logger.error(
                f"检查集合 '{collection_name}' 是否存在时出错: {e}", exc_info=True
            )
            return False

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        if not self.client:
            raise ConnectionError("ChromaDB 客户端未初始化。")

        if not await self.collection_exists(collection_name):
            logger.warning(f"ChromaDB 集合 '{collection_name}' 不存在，将自动创建。")
            await self.create_collection(collection_name)

        collection = await asyncio.to_thread(
            self.client.get_collection, name=collection_name
        )
        if collection_name not in self.doc_mappings:
            self.doc_mappings[collection_name] = {}

        processing_queue: asyncio.Queue[ProcessingBatch] = asyncio.Queue()
        all_added_ids: List[str] = []

        # --- 生产者：将文档分批放入队列 ---
        num_batches = 0
        for i in range(0, len(documents), self.embedding_batch_size):
            batch_docs = documents[i : i + self.embedding_batch_size]
            await processing_queue.put(ProcessingBatch(documents=batch_docs))
            num_batches += 1
        logger.info(f"已将 {len(documents)} 份文档分成 {num_batches} 个批次放入队列。")

        # --- 消费者：从队列取出批次处理，包含重试逻辑 ---
        processed_batches_count = 0
        failed_batches_discarded_count = 0

        while processed_batches_count < num_batches:
            try:
                processing_batch = await processing_queue.get()
            except asyncio.CancelledError:
                break

            current_docs = processing_batch.documents
            current_retry = processing_batch.retry_count

            log_prefix = f"[批次, 重试 {current_retry}/{MAX_RETRIES}]"
            logger.debug(f"{log_prefix} 正在处理 {len(current_docs)} 个文档...")

            try:
                # --- 核心处理逻辑 ---
                texts = [doc.text_content for doc in current_docs]
                embeddings = await self.embedding_util.get_embeddings_async(
                    texts, collection_name
                )

                valid_docs, doc_ids, metadatas, embeddings_to_add = [], [], [], []
                for i, doc in enumerate(current_docs):
                    if embeddings and embeddings[i]:
                        doc_id = str(uuid.uuid4())
                        doc.id = doc_id
                        valid_docs.append(doc)
                        doc_ids.append(doc_id)
                        metadatas.append(doc.metadata)
                        embeddings_to_add.append(embeddings[i])

                if doc_ids:
                    # 使用 to_thread 执行同步的 add 操作
                    await asyncio.to_thread(
                        collection.add,
                        embeddings=embeddings_to_add,
                        metadatas=metadatas,
                        documents=[d.text_content for d in valid_docs],
                        ids=doc_ids,
                    )
                    for doc in valid_docs:
                        self.doc_mappings[collection_name][doc.id] = doc
                    all_added_ids.extend(doc_ids)
                    logger.debug(f"{log_prefix} 成功添加 {len(doc_ids)} 个文档。")
                else:
                    logger.warning(f"{log_prefix} 没有有效的文档可供添加。")

                processed_batches_count += 1
                processing_queue.task_done()

            except Exception as e:
                logger.error(f"{log_prefix} 处理失败: {e}", exc_info=True)
                if current_retry < MAX_RETRIES:
                    processing_batch.retry_count += 1
                    await processing_queue.put(processing_batch)
                    logger.warning(f"{log_prefix} 将批次重新放入队列进行重试...")
                else:
                    logger.error(f"{log_prefix} 批次达到最大重试次数，将被丢弃。")
                    processed_batches_count += 1  # 丢弃也算处理完成
                    failed_batches_discarded_count += 1
                    processing_queue.task_done()

        await processing_queue.join()

        logger.info(
            f"向 ChromaDB 集合 '{collection_name}' 添加操作完成。成功添加 {len(all_added_ids)} 个文档。"
        )
        if failed_batches_discarded_count > 0:
            logger.warning(
                f"其中 {failed_batches_discarded_count} 个批次因重试失败被丢弃。"
            )
        return all_added_ids

    async def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        if not self.client or not await self.collection_exists(collection_name):
            return []

        collection = await asyncio.to_thread(
            self.client.get_collection, name=collection_name
        )
        if await asyncio.to_thread(collection.count) == 0:
            return []

        query_embedding = await self.embedding_util.get_embedding_async(
            query_text, collection_name
        )
        if not query_embedding:
            return []

        try:
            # 同步操作，使用 to_thread
            results = await asyncio.to_thread(
                collection.query,
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            docs = []
            if results and results.get("ids") and results["ids"][0]:
                ids, distances, metadatas, contents = (
                    results["ids"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                    results["documents"][0],
                )

                for i, doc_id in enumerate(ids):
                    similarity = 1.0 / (1.0 + distances[i])
                    doc = Document(
                        id=doc_id, text_content=contents[i], metadata=metadatas[i]
                    )
                    docs.append((doc, similarity))
            return docs
        except Exception as e:
            logger.error(
                f"在 ChromaDB 集合 '{collection_name}' 中搜索失败: {e}",
                exc_info=True,
            )
            return []

    async def delete_collection(self, collection_name: str) -> bool:
        if not self.client or not await self.collection_exists(collection_name):
            return False
        try:
            await asyncio.to_thread(self.client.delete_collection, name=collection_name)
            if collection_name in self.doc_mappings:
                del self.doc_mappings[collection_name]
            logger.info(f"ChromaDB 集合 '{collection_name}' 删除成功。")
            return True
        except Exception as e:
            logger.error(
                f"删除 ChromaDB 集合 '{collection_name}' 时出错: {e}",
                exc_info=True,
            )
            return False

    async def list_collections(self) -> List[str]:
        if not self.client:
            return []
        try:
            collections = await asyncio.to_thread(self.client.list_collections)
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"列出 ChromaDB 集合失败: {e}", exc_info=True)
            return []

    async def count_documents(self, collection_name: str) -> int:
        if not self.client or not await self.collection_exists(collection_name):
            return 0
        try:
            collection = await asyncio.to_thread(
                self.client.get_collection, name=collection_name
            )
            return await asyncio.to_thread(collection.count)
        except Exception as e:
            logger.error(
                f"统计 ChromaDB 集合 '{collection_name}' 文档数时出错: {e}",
                exc_info=True,
            )
            return 0

    async def close(self):
        logger.info("ChromaDB 客户端无需显式关闭。")
        await asyncio.sleep(0)
