import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
load_dotenv(dotenv_path="scripts/.env")

VECTOR_DB_DIR = "vector_db" # Path relative to where the API is run (root)

class RAGService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set.")

        # 1. Initialize Embedding Model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )

        # 2. Load the Vector Database
        self.vectordb = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

        # 3. Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-pro-latest", # <--- THIS IS THE FIX
            google_api_key=self.api_key,
            temperature=0.3
        )

        # 4. Define the RAG Prompt Template
        rag_prompt_template = """
        You are "LawGeeks", a specialized AI assistant. Your goal is to answer a user's specific question about their legal document, using *only* the provided context.
        
        You have been given three pieces of information:
        1.  **THE USER'S DOCUMENT**: The full text of their agreement.
        2.  **RELEVANT LEGAL CONTEXT**: Snippets from Indian law (e.g., The Contract Act, RERA) that are relevant to the user's question.
        3.  **THE USER'S QUESTION**: The specific question the user asked.

        **INSTRUCTIONS:**
        1.  First, analyze the **USER'S DOCUMENT** to find clauses that relate to the **USER'S QUESTION**.
        2.  Next, use the **RELEVANT LEGAL CONTEXT** to understand the standard legal position or definitions.
        3.  Combine these insights to provide a clear, simple, and direct answer.
        4.  If the user's document is silent on the issue, say so.
        5.  If the user's document *contradicts* the legal context, point this out (e.g., "Your document states X, which is unusual as the standard legal position is Y...").
        6.  **DO NOT** make up information. If the answer cannot be found in the provided texts, state that you cannot answer.
        7.  **DO NOT** provide legal advice. Frame your answer as "This clause appears to mean..." or "This document states...".

        ---
        **THE USER'S DOCUMENT:**
        {document_context}
        ---
        **RELEVANT LEGAL CONTEXT:**
        {rag_context}
        ---
        **THE USER'S QUESTION:**
        {question}
        ---

        **Your Answer:**
        """
        self.rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
        
        # 5. Create the RAG Chain (using LCEL)
        self.rag_chain = (
            {
                "rag_context": self.retriever | self._format_docs, 
                "question": RunnablePassthrough(),
                "document_context": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        # Helper to join retrieved docs into a single string
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def answer_user_query(self, document_text: str, user_question: str) -> str:
        """
        Answers a user's question using the RAG pipeline.
        """
        try:
            # We pass a dict to the chain that matches the keys in the first part
            # of the chain. "document_context" is passed straight through.
            # "question" is passed straight through.
            # The retriever is called *with* the "question" to get "rag_context".
            
            # NOTE: LangChain's retriever pipeline is invoked with the entire input dict.
            # By default, the retriever's input is the "question" key.
            # We need to pass the *other* keys through manually.
            
            rag_chain = (
                {
                    "rag_context": self.retriever,
                    "question": RunnablePassthrough(),
                    "document_context": RunnablePassthrough() # Pass document_context through
                }
                | RunnablePassthrough.assign(
                    rag_context=lambda x: self._format_docs(x["rag_context"]),
                    document_context=lambda x: x["document_context"] # Ensure it's passed
                )
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Invoke the chain
            # The input to the retriever will be the 'question'
            # The other keys ('document_context') will be passed along
            response = rag_chain.invoke({
                "question": user_question, 
                "document_context": document_text
            })
            return response
            
        except Exception as e:
            print(f"Error in RAG chain: {e}")
            return "I encountered an error trying to find the answer. Please try rephrasing your question."