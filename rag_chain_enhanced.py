"""
RAG Chain - Using Groq SDK Directly (No LangChain ChatGroq)
Bypasses all compatibility issues
"""

from typing import List, Dict, Any, Optional
import os
import re
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class ConversationalRAGChain:
    """
    RAG chain using Groq SDK directly - No LangChain wrapper issues
    """

    def __init__(self, vector_store, groq_api_key: str, model_name: str = "llama-3.1-8b-instant"):
        """Initialize with Groq SDK directly"""
        self.vector_store = vector_store
        self.model_name = model_name

        # Use Groq SDK directly - no ChatGroq wrapper
        self.client = Groq(api_key=groq_api_key)
        print("✅ Groq client initialized successfully")

    def _needs_contextualization(self, question: str) -> bool:
        """Detect if a question needs conversation context"""
        word_indicators = [
            r'\bit\b', r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b',
        ]

        phrase_indicators = [
            'differ', 'difference', 'compare', 'comparison',
            'example', 'more about', 'also', 'additionally',
            'what about', 'how about', 'the same', 'similar',
            'mentioned', 'said', 'explained', 'previous', 'earlier'
        ]

        question_lower = question.lower()

        for pattern in word_indicators:
            if re.search(pattern, question_lower):
                return True

        for indicator in phrase_indicators:
            if indicator in question_lower:
                return True

        words = question.split()
        if len(words) <= 3:
            question_starters = ['what', 'when', 'where', 'who', 'why', 'how', 'which']
            if words and words[0].lower() not in question_starters:
                return True
            if len(words) <= 2:
                return True

        return False

    def _retrieve_relevant_docs(self, query: str, k: int = 12) -> List[Document]:
        """Retrieve relevant documents from vector store"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []

    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for inclusion in prompt"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            metadata = doc.metadata

            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')

            formatted.append(
                f"[Document {i}]\n"
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content: {content}\n"
            )

        return "\n".join(formatted)

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        k: int = 12
    ) -> Dict[str, Any]:
        """Query with Groq SDK directly"""
        if chat_history is None:
            chat_history = []

        # Format history
        formatted_history = "This is the first question in our conversation."
        if chat_history:
            history_lines = []
            for msg in chat_history:
                role = msg['role'].capitalize()
                content = msg['content']
                history_lines.append(f"{role}: {content}")
            formatted_history = "\n".join(history_lines)

        needs_context = self._needs_contextualization(question) and len(chat_history) > 0
        search_query = question

        # Retrieve documents
        relevant_docs = self._retrieve_relevant_docs(search_query, k=k)
        context = self._format_docs(relevant_docs)

        # Build prompt
        system_prompt = """You are a friendly and knowledgeable ISO 26262 functional safety expert.
Your goal is to help automotive engineers and safety professionals understand and apply the standard.

Write in a conversational, helpful tone:
- Use "you" to address the reader
- Break down complex concepts into clear explanations  
- Provide practical examples when relevant
- Be concise but thorough
- Use everyday language while maintaining technical accuracy"""

        user_message = f"""Context from ISO 26262:
{context}

Conversation History:
{formatted_history}

Question: {question}

Answer in a warm, professional, and helpful manner:"""

        try:
            # Call Groq API directly
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            answer = response.choices[0].message.content

        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            answer = "I encountered an error while generating the answer. Please try again."

        # Extract sources
        sources = []
        for doc in relevant_docs:
            source_info = {
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'content_preview': doc.page_content[:200] + '...'
            }
            sources.append(source_info)

        return {
            'answer': answer,
            'sources': sources,
            'original_question': question,
            'contextualized_question': search_query if needs_context else None,
            'used_context': needs_context
        }


class RAGChain(ConversationalRAGChain):
    """Backward compatible wrapper"""
    def simple_query(self, question: str, k: int = 12) -> Dict[str, Any]:
        return self.query(question, chat_history=[], k=k)