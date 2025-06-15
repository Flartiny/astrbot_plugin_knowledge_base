import chromadb
import uuid
from typing import List, Dict, Tuple, Optional, Any
from .base import (
    VectorDBBase,
    Document,
    DEFAULT_BATCH_SIZE
)
from astrbot.api import logger
from ..utils.embedding import EmbeddingSolutionHelper
import asyncio


class ChromaStore(VectorDBBase):
    def __init__(
        self, embedding_util: EmbeddingSolutionHelper, data_path: str, **kwargs
    ):
        super().__init__(embedding_util, data_path)
        self.client: Optional[chromadb.Client] = None
        self.doc_mappings: Dict[str, Dict[str, Document]] = {}

    async def initialize(self):
        logger.info(f"Initializing ChromaDB client at path: {self.data_path}...")
        try:
            self.client = chromadb.PersistentClient(path=self.data_path)
            logger.info("ChromaDB client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            self.client = None
            raise

    async def create_collection(self, collection_name: str):
        if not self.client:
            raise ConnectionError("ChromaDB client not initialized.")
        try:
            self.client.create_collection(name=collection_name)
            self.doc_mappings[collection_name] = {}
            logger.info(
                f"ChromaDB collection '{collection_name}' created successfully."
            )
        except chromadb.errors.DuplicateCollectionError:
            logger.info(f"ChromaDB collection '{collection_name}' already exists.")
        except Exception as e:
            logger.error(
                f"Failed to create ChromaDB collection '{collection_name}': {e}",
                exc_info=True,
            )
            raise

    async def collection_exists(self, collection_name: str) -> bool:
        if not self.client:
            return False
        try:
            collections = self.client.list_collections()
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            logger.error(
                f"Error checking if collection '{collection_name}' exists: {e}",
                exc_info=True,
            )
            return False

    async def add_documents(
        self, collection_name: str, documents: List[Document]
    ) -> List[str]:
        if not self.client:
            raise ConnectionError("ChromaDB client not initialized.")

        if not await self.collection_exists(collection_name):
            await self.create_collection(collection_name)

        collection = self.client.get_collection(name=collection_name)
        if collection_name not in self.doc_mappings:
            self.doc_mappings[collection_name] = {}

        all_doc_ids = []

        # Process documents in batches
        for i in range(0, len(documents), DEFAULT_BATCH_SIZE):
            batch_docs = documents[i : i + DEFAULT_BATCH_SIZE]
            logger.info(
                f"Processing batch {i // DEFAULT_BATCH_SIZE + 1} with {len(batch_docs)} documents."
            )

            texts = [doc.text_content for doc in batch_docs]
            embeddings = await self.embedding_util.get_embeddings_async(
                texts, collection_name
            )

            valid_docs = []
            doc_ids = []
            metadatas = []
            embeddings_to_add = []

            for j, doc in enumerate(batch_docs):
                if embeddings and embeddings[j]:
                    doc_id = str(uuid.uuid4())
                    doc.id = doc_id
                    doc.embedding = embeddings[j]

                    valid_docs.append(doc)
                    doc_ids.append(doc_id)
                    metadatas.append(doc.metadata)
                    embeddings_to_add.append(embeddings[j])

            if not doc_ids:
                logger.warning("No valid documents or embeddings in the current batch.")
                continue

            try:
                collection.add(
                    embeddings=embeddings_to_add,
                    metadatas=metadatas,
                    documents=[doc.text_content for doc in valid_docs],
                    ids=doc_ids,
                )

                for doc in valid_docs:
                    self.doc_mappings[collection_name][doc.id] = doc

                all_doc_ids.extend(doc_ids)
                logger.info(
                    f"Successfully added batch of {len(doc_ids)} documents to ChromaDB collection '{collection_name}'."
                )

            except Exception as e:
                logger.error(
                    f"Failed to add a batch of documents to ChromaDB collection '{collection_name}': {e}",
                    exc_info=True,
                )
                # Decide if you want to stop or continue with other batches
                # For now, we log the error and continue
                pass

        return all_doc_ids

    async def search(
        self, collection_name: str, query_text: str, top_k: int = 5
    ) -> List[Tuple[Document, float]]:
        if not self.client or not await self.collection_exists(collection_name):
            return []

        collection = self.client.get_collection(name=collection_name)
        if collection.count() == 0:
            return []

        query_embedding = await self.embedding_util.get_embedding_async(
            query_text, collection_name
        )
        if not query_embedding:
            return []

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            docs = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                distances = results["distances"][0]
                metadatas = results["metadatas"][0]
                contents = results["documents"][0]

                for i, doc_id in enumerate(ids):
                    similarity = 1.0 / (1.0 + distances[i])
                    doc = Document(
                        id=doc_id, text_content=contents[i], metadata=metadatas[i]
                    )
                    docs.append((doc, similarity))
            return docs
        except Exception as e:
            logger.error(
                f"Failed to search in ChromaDB collection '{collection_name}': {e}",
                exc_info=True,
            )
            return []

    async def delete_collection(self, collection_name: str) -> bool:
        if not self.client or not await self.collection_exists(collection_name):
            return False
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self.doc_mappings:
                del self.doc_mappings[collection_name]
            logger.info(
                f"ChromaDB collection '{collection_name}' deleted successfully."
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to delete ChromaDB collection '{collection_name}': {e}",
                exc_info=True,
            )
            return False

    async def list_collections(self) -> List[str]:
        if not self.client:
            return []
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list ChromaDB collections: {e}", exc_info=True)
            return []

    async def count_documents(self, collection_name: str) -> int:
        if not self.client or not await self.collection_exists(collection_name):
            return 0
        try:
            collection = self.client.get_collection(name=collection_name)
            return collection.count()
        except Exception as e:
            logger.error(
                f"Failed to count documents in ChromaDB collection '{collection_name}': {e}",
                exc_info=True,
            )
            return 0

    async def close(self):
        logger.info("ChromaDB client does not require explicit closing.")
        await asyncio.sleep(0)
