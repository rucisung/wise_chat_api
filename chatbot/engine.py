from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain, StuffDocumentsChain, LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from .document_loader import DocumentProcessor
import logging
import os
from collections import defaultdict

# 로그 디렉토리 생성
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'chatbot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class InMemoryHistory:
    def __init__(self):
        self.store = defaultdict(list)

    def get_session_messages(self, session_id: str) -> list[BaseMessage]:
        return self.store[session_id]

    def add_session_message(self, session_id: str, message: BaseMessage):
        self.store[session_id].append(message)

    def clear_session(self, session_id: str):
        self.store[session_id] = []

class SessionSpecificChatHistory(BaseChatMessageHistory):
    def __init__(self, history_manager: InMemoryHistory, session_id: str):
        self.history_manager = history_manager
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieve messages for the session, ordered oldest to newest."""
        return self.history_manager.get_session_messages(self.session_id)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the session history."""
        for message in messages:
            self.history_manager.add_session_message(self.session_id, message)

    def clear(self) -> None:
        """Clear all messages from the session history."""
        self.history_manager.clear_session(self.session_id)

class ChatbotEngine:
    def __init__(self, model_name="gemini-1.5-pro", api_key=None, doc_processor=None):
        self.llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
        self.history_manager = InMemoryHistory()
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.vector_store = None
        self.retrieval_chain = None
        self.conversation_chain = None
        self.doc_processor = doc_processor or DocumentProcessor(api_key=api_key)
        self.role = "default"
        self.role_source = "initial"
        self.role_prompts = {
            "default": "You are a helpful and friendly assistant providing accurate and concise answers.",
            "tech_support": "You are a professional technical support specialist. Provide detailed, accurate, and step-by-step technical guidance.",
            "educator": "You are a knowledgeable educator. Explain concepts clearly and concisely, using examples suitable for learners.",
            "analyst": "You are a data analyst. Provide precise, data-driven insights and avoid speculation."
        }
        self.intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="Analyze the following user query and classify the intent to determine the most suitable expert role (tech_support, educator, analyst, or default). Return only the role name.\\nQuery: {query}"
        )
        self._update_conversation_chain_base()
        logger.info("ChatbotEngine initialized with default role")

    def _get_base_chat_prompt_template(self):
        """Gets the prompt template for the base conversation chain."""
        system_prompt = self.role_prompts.get(self.role, self.role_prompts["default"])
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return prompt

    def _get_rag_combine_prompt_template(self):
        """
        Gets the prompt template for combining retrieved documents and question within the RAG chain.
        This template must accept 'context', 'chat_history', and 'question' variables.
        """
        system_prompt_text = self.role_prompts.get(self.role, self.role_prompts["default"]) + \
                             "\nUse the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer."
        system_prompt_text += " Cite the source document names if possible."

        rag_combine_template_string = """{system_prompt}
--------------------
{context}
--------------------
Chat History:
{chat_history}
Human: {question}"""

        prompt = ChatPromptTemplate.from_template(rag_combine_template_string).partial(
             system_prompt=system_prompt_text
        )
        return prompt

    def _update_conversation_chain_base(self):
        """Updates the base conversation chain (used when RAG is not active or 'base' mode is requested)."""
        chain = self._get_base_chat_prompt_template() | self.llm
        self.conversation_chain = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=lambda session_id: SessionSpecificChatHistory(self.history_manager, session_id),
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        logger.info("Base conversation chain updated.")

    def set_role(self, role):
        if role not in self.role_prompts:
            logger.warning(f"Invalid role attempted: {role}")
            return False
        self.role = role
        self.role_source = "set_role"
        self._update_conversation_chain_base()
        if self.retrieval_chain:
            self.process_documents([])
        logger.info(f"Role set to {role} via set_role")
        return True

    def auto_set_role(self, query):
        if not query.strip():
            self.role = "default"
            self.role_source = "auto_role_empty"
            logger.info("Role set to default due to empty query")
            self._update_conversation_chain_base()
            if self.vector_store:
                logger.info(f"RAG chain exists, attempting to re-create with new role prompt after auto-setting role.")
                try:
                    self.retrieval_chain = ConversationalRetrievalChain.from_llm(
                        llm=self.llm,
                        retriever=self.vector_store.as_retriever(),
                        memory=self.memory,
                        combine_docs_chain_kwargs={"prompt": self._get_rag_combine_prompt_template()},
                        return_source_documents=True
                    )
                    logger.info("RAG chain re-created with new role prompt after auto-setting role.")
                except Exception as e:
                    logger.error(f"Failed to re-create RAG chain with new role prompt after auto-setting role: {e}")
            return self.role
        intent_chain = self.intent_prompt | self.llm
        role = intent_chain.invoke({"query": query}).content.strip()
        if role in self.role_prompts:
            self.set_role(role)
            self.role_source = "auto_role"
            logger.info(f"Role set to {role} via auto_role for query: {query}")
        else:
            self.set_role("default")
            self.role_source = "auto_role_fallback"
            logger.info(f"Role set to default via auto_role fallback for query: {query}")
        return self.role

    def get_role_info(self):
        return {"role": self.role, "source": self.role_source}

    def process_documents(self, files):
        chunks = self.doc_processor.process_files(files)
        if chunks:
            self.vector_store = Chroma.from_documents(chunks, self.doc_processor.embeddings)
            self.retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": self._get_rag_combine_prompt_template()},
                return_source_documents=True
            )
            logger.info(f"Documents processed, vector store created with {len(chunks)} chunks")
        else:
            self.vector_store = None
            self.retrieval_chain = None
            logger.info("No documents processed, vector store cleared")
            self._update_conversation_chain_base()

    def get_response(self, query, mode: str = "auto"):
        """
        Gets a response to the query based on the specified mode.
        Returns a dictionary with 'answer', 'sources', and 'mode_used'.
        'sources' will be a list of file paths used for RAG, or empty if RAG was not used.

        Args:
            query: User's query string.
            mode: 'auto', 'rag', or 'base'.
                  'auto': Use RAG if documents are processed, otherwise use base chat.
                  'rag': Use RAG if available. If not, indicate error.
                  'base': Use base chat, ignoring RAG even if available.
        """
        if not query.strip():
            logger.warning("Empty query received")
            return {"answer": "Query cannot be empty", "sources": [], "mode_used": "none"}

        session_id = "default"
        used_mode = "base"

        chain_to_use = None
        if mode == "rag":
            if self.retrieval_chain:
                chain_to_use = self.retrieval_chain
                used_mode = "rag"
                logger.info(f"Explicitly using RAG chain for query: {query}")
            else:
                logger.warning(f"RAG mode requested but no documents processed for query: {query}")
                return {"answer": "RAG mode was requested but no documents have been processed.", "sources": [], "mode_used": "error_rag_unavailable"}

        elif mode == "base":
             chain_to_use = self.conversation_chain
             used_mode = "base"
             logger.info(f"Explicitly using base conversation chain for query: {query}")

        elif mode == "auto":
            if self.retrieval_chain:
                chain_to_use = self.retrieval_chain
                used_mode = "rag"
                logger.info(f"Automatically using RAG chain for query: {query}")
            else:
                chain_to_use = self.conversation_chain
                used_mode = "base"
                logger.info(f"Automatically using base conversation chain for query: {query}")
        else:
            logger.warning(f"Unknown mode '{mode}' requested. Falling back to auto mode.")
            if self.retrieval_chain:
                chain_to_use = self.retrieval_chain
                used_mode = "rag"
                logger.info(f"Automatically using RAG chain for query: {query}")
            else:
                chain_to_use = self.conversation_chain
                used_mode = "base"
                logger.info(f"Automatically using base conversation chain for query: {query}")

        try:
            if used_mode == "rag":
                rag_response = chain_to_use.invoke({"question": query})
                logger.info(f"Response generated with role {self.role} ({used_mode} mode) for query: {query}")

                answer = rag_response.get("answer", "No answer found.")
                source_documents = rag_response.get("source_documents", [])

                sources = []
                if source_documents:
                    source_paths = set()
                    for doc in source_documents:
                        if doc.metadata and 'source' in doc.metadata:
                            source_paths.add(doc.metadata['source'])
                    sources = list(source_paths)

                logger.info(f"Sources used for RAG: {sources}")
                return {"answer": answer, "sources": sources, "mode_used": used_mode}

            elif used_mode == "base":
                 response_message = chain_to_use.invoke(
                    {"input": query},
                    config={"configurable": {"session_id": session_id}}
                 )
                 if hasattr(response_message, 'content'):
                    return {"answer": response_message.content, "sources": [], "mode_used": used_mode}
                 else:
                    logger.error(f"Unexpected response type from base chain: {type(response_message)}. Full response: {response_message}")
                    return {"answer": str(response_message) if response_message is not None else "Error: Could not process response.", "sources": [], "mode_used": used_mode}

            else:
                 return {"answer": "Internal error: No chain selected.", "sources": [], "mode_used": "error_internal"}

        except Exception as e:
            logger.error(f"Error during chain invocation (mode: {used_mode}) for query '{query}': {e}")
            return {"answer": f"An error occurred during processing ({used_mode} mode): {e}", "sources": [], "mode_used": "error_invocation"}