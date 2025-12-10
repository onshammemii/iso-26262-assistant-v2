
#  ISO 26262 Safety Assistant

An AI-powered assistant that helps engineers understand and implement **ISO 26262 Functional Safety** standards using advanced Retrieval-Augmented Generation (RAG) technology.

##  Features

- **AI-Powered Q&A**: Ask questions about ISO 26262 and get instant, accurate answers
- **Multi-turn Conversations**: Maintain context across multiple questions
- **Semantic Search**: Find relevant safety documentation using vector embeddings
- **Expert Knowledge**: Trained on comprehensive ISO 26262 documentation
- **Fast & Reliable**: Real-time responses with source attribution

##  Use Cases

- **Functional Safety Engineers** - Quick reference for safety standard requirements
- **Automotive Teams** - Understand ASIL levels and safety integrity levels
- **Compliance Officers** - Verify safety processes and documentation
- **Students/Trainees** - Learn functional safety concepts interactively

##  Getting Started

### Try it Live
Simply ask questions in the chat interface! No installation needed.

### Example Questions

- "What is ISO 26262 and why is it important?"
- "How do I determine the ASIL level for my project?"
- "What are the requirements for ASIL D?"
- "Explain the V-Model in functional safety"
- "What testing methods are required for HSI?"

##  Architecture

```
User Input
    ↓
Vector Store (Semantic Search)
    ↓
RAG Chain (Context Retrieval)
    ↓
LLM (Groq API - llama-3.1-8b-instant)
    ↓
Answer Generation with Sources
```

##  Tech Stack

- **Backend**: Flask 3.1.2
- **LLM Framework**: LangChain 0.1.9
- **Vector Database**: FAISS 1.13.1
- **Embeddings**: Sentence Transformers 5.1.2
- **LLM Provider**: Groq API (llama-3.1-8b-instant)
- **Deployment**: Docker + Hugging Face Spaces
- **Frontend**: HTML5 + JavaScript

##  Key Components

### RAG System
- **Retrieval**: Uses semantic search to find relevant ISO 26262 documents
- **Augmentation**: Combines user question with retrieved context
- **Generation**: LLM generates accurate answers based on context

### Conversation Management
- Multi-turn conversation history
- Context-aware responses
- Session persistence

##  About ISO 26262

ISO 26262 is the international standard for functional safety in automotive electrical/electronic systems. It defines:

- **ASIL Levels** (A, B, C, D): Safety Integrity Levels
- **V-Model**: Development and verification process
- **FMEA/FTA**: Failure analysis methods
- **Testing Requirements**: HSI, integration, system testing
- **Documentation**: Safety plans and reports

##  How It Works

1. **Your Question** → Converted to semantic embeddings
2. **Vector Search** → Finds relevant ISO 26262 content
3. **RAG Retrieval** → Pulls context from documentation
4. **LLM Processing** → Generates accurate, cited answer
5. **Response** → Delivered with source references

##  Privacy & Data

-  Conversations stored locally (session-based)
-  No personal data collection
-  Queries sent only to Groq API
-  Source documents referenced but not stored in chat

##  Limitations

- Responses based on trained knowledge cutoff
- Complex technical decisions should involve safety experts
- Not a substitute for official ISO 26262 documentation
- For critical safety systems, consult regulatory bodies

##  Contributing

Found an issue? Have suggestions?
- Create an issue or pull request
- Share feedback in the Community tab

##  License

This project is licensed under the MIT License - see LICENSE file for details.

##  Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Groq API](https://groq.com/)
- Hosted on [Hugging Face Spaces](https://huggingface.co/spaces)
- Embeddings by [Sentence Transformers](https://www.sbert.net/)

##  Support

- **Issues**: GitHub Issues
- **Feedback**: HF Space Community tab
- **Questions**: Check example cards above

---

**Last Updated**: December 2025  
**Status**: Active & Maintained  
**Version**: 2.0
