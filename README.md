# ANLP_Final_Project
#### Task_01
# Medical RAG Assistant with LangChain Integration
# dataset link: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
A Retrieval-Augmented Generation (RAG) system for medical question answering, built with clinical transcription data and LangChain patterns.

##  What It Does

This system helps answer medical questions by:
1. **Searching** through thousands of medical transcriptions using FAISS
2. **Finding** relevant medical information with semantic search
3. **Generating** accurate, evidence-based answers using Gemini
4. **Using** LangChain patterns for clean integration
5. **Providing** source citations for transparency

##  LangChain Integration

This project uses **LangChain** as required by the assignment through:

### **LangChain Components:**
- **`RunnableLambda`** from `langchain-core` - Wraps Gemini API calls in LangChain pattern
- **LangChain invocation pattern** - `.invoke()` method for consistent LLM calls
- **Medical prompt engineering** - Structured prompts for medical Q&A

### **Why This Approach?**
- **Stability**: Avoids dependency conflicts with `langchain-google-genai`
- **Simplicity**: Direct API calls wrapped in LangChain patterns
- **Meets Requirements**: Uses core LangChain components as specified
- **Efficiency**: Combines best of both worlds

##  System Details

- **Medical Data**: 3,898 clinical transcription records
- **Text Chunks**: 29,713 processed medical text segments
- **Specialties**: 39 different medical specialties covered
- **Search Technology**: FAISS vector similarity search
- **AI Model**: Google Gemini 2.0 Flash via LangChain pattern
- **Framework**: Streamlit for web interface

##  Quick Start

### 1. Get API Key
- Visit [Google AI Studio](https://aistudio.google.com/)
- Sign in with Google account
- Click "Get API Key" and create a new key
- Copy your API key

### 2. Local Installation
```bash
# Clone repository
git clone <repository-url>
cd medical-rag-assistant

# Install requirements
pip install -r requirements.txt

# Run the Streamlit app
streamlit run medical_rag_app.py
```

### 3. Google Colab Setup
This project can be run directly in Google Colab. Ensure your Google Drive is mounted to access the dataset and store model artifacts.

### 4. Deployment to Streamlit Cloud
1. Upload all project files (including `medical_rag_app.py`, `requirements.txt`, and the `medical_rag_system` directory) to a GitHub repository.
2. Go to [Streamlit Share](https://share.streamlit.io/) and connect your repository.
3. Deploy the app. Make sure to add your Gemini API key as a secret in Streamlit Cloud (`st.secrets.GEMINI_API_KEY`).

##  Project Structure

- `medical_rag_app.py`: The main Streamlit application.
- `requirements.txt`: Python dependencies.
- `medical_rag_system/`: Directory containing saved FAISS index, chunk metadata, and embeddings.
- `dataset_task01.zip`: Raw medical transcription dataset (expected in Google Drive).

##  Disclaimer

This system is for educational and demonstrative purposes only and should **not** be used for actual medical advice or diagnosis. Always consult with a qualified healthcare professional for any medical concerns.


README content generated and displayed successfully!

Key information:
• LangChain `RunnableLambda` for Gemini integration
• Quick Start instructions for local and Colab setup
• Deployment guide for Streamlit Cloud

#### Task02
# Policy Compliance Checker RAG System (Task 02)
# dataset link: https://www.atticusprojectai.org/cuad	

## Project Overview
This project implements a Policy Compliance Checker RAG (Retrieval-Augmented Generation) System designed to automate contract compliance analysis. It leverages advanced Natural Language Processing (NLP) techniques, including vector embeddings, a robust vector store, and a generative AI model (Gemini) integrated through custom LangChain tools, all orchestrated by an intelligent multi-step agent workflow.

## Architectural Components

The system is comprised of several key components:

1.  **Compliance Rules**: A predefined set of 18 compliance rules, inspired by CUAD categories, against which contracts are evaluated. Each rule includes a description, compliance requirement, severity, and suggested remediation.
2.  **Text Processing (Chunking)**: Raw contract text is processed into smaller, manageable `chunks` and `legal sections` using `RecursiveCharacterTextSplitter` to facilitate efficient retrieval. This includes both general text segments and specifically extracted legal clauses.
3.  **Vector Store**: A `FAISS` (Facebook AI Similarity Search) vector store is used to store high-dimensional `embeddings` of the processed contract chunks. The `all-MiniLM-L6-v2` SentenceTransformer model generates these 384-dimensional embeddings, enabling rapid semantic search.
4.  **Custom LangChain Tools**: Specialized functions are wrapped as LangChain tools, allowing an AI agent to interact with the system's core functionalities. These include tools for checking single or multiple rules, retrieving relevant contract sections, and comparing compliance between contracts.
5.  **Multi-Step Agent Workflow**: An intelligent `Agent Workflow` processes multi-step compliance questions. It dynamically analyzes query types (e.g., compliance check, retrieval, comparison, comprehensive analysis) and executes the appropriate LangChain tools, integrating RAG for coherent, informative responses.

## Achievements and Successful Implementations

Despite facing API quota challenges with the Gemini model during development, the core architecture and many functional components were successfully implemented and demonstrated:

*   **End-to-End RAG System Architecture**: A foundational integrated RAG system was successfully designed and implemented, covering data ingestion, chunking, vector storage, LLM integration, LangChain tooling, and an intelligent agent.
*   **Effective Text Processing and Chunking**: The `ContractTextProcessor` demonstrated robust text preparation capabilities by successfully creating manageable text chunks and extracting legal sections from contract data.
*   **Robust Vector Store and Embeddings**: The `ComplianceVectorStore` was successfully initialized, creating high-dimensional embeddings and building an efficient FAISS index for semantic search.
*   **Functional Retrieval Mechanism**: The `retrieve_contract_sections` tool proved capable of accurately retrieving relevant contract sections based on semantic queries, validating a core RAG component.
*   **Well-Structured Custom LangChain Tools**: Custom LangChain `BaseTool` implementations were successfully defined and integrated with correctly configured input schemas, enabling clear communication within the agent framework.
*   **Intelligent Agent Logic for Query Routing**: The `ComplianceAgent` demonstrated intelligent routing by accurately classifying user queries and invoking appropriate custom tools for multi-step reasoning.
*   **Comprehensive System Integration and Report Generation**: The `PolicyComplianceRAGSystem` successfully integrated all components, showcasing its ability to orchestrate complex tasks, generate structured reports, and produce visualizations, even when LLM analysis was limited by external factors.

## Limitations and Next Steps

### Primary Limitation: Gemini API Rate Limiting

The most significant limitation encountered was persistent `429 POST` errors from the Gemini API, indicating rate limiting or quota issues. This severely hampered the real-time compliance checking functionality, leading to `ERROR` statuses and 0 scores for LLM-dependent analyses. This impacts reliability, slows testing, and poses scalability concerns.

#### Potential Solutions and Workarounds:
*   **Optimize API Calls**: Reduce frequency and combine queries where feasible.
*   **Implement Retry Mechanism**: Build robust retry logic with exponential backoff.
*   **Explore Alternative Models**: Investigate other LLM providers (e.g., OpenAI, Anthropic) or open-source models (Llama, Mistral, Gemma).
*   **Local Model Deployment**: Consider deploying smaller, fine-tuned LLMs locally for high-throughput scenarios.
*   **Batch Processing**: Implement batching for non-real-time analyses.
*   **Increase Quota**: Apply for higher API quotas from Google Cloud.

### Other Limitations and Future Enhancements:

1.  **Dataset Scope**: The current implementation was primarily tested with `CUAD_v1_README.txt` due to issues with extracting other contract types. **Next Step**: Ensure robust extraction and loading of diverse contract documents (PDF, DOCX, TXT) from the full CUAD dataset.
2.  **Parsing Robustness**: The `_parse_compliance_response` method in `ComplianceCheckerTools` sometimes experienced issues like `KeyError: 'risk_level'` due to inconsistent LLM output formats. **Next Step**: Enhance parsing logic to be more resilient to variations in LLM responses, possibly using JSON output parsers or more flexible regex.
3.  **Scalability for Large Contracts**: While chunking is implemented, large contract texts might still hit token limits. **Next Step**: Optimize chunk selection for compliance checks by using retrieval to provide only the *most relevant* sections to the LLM.
4.  **Refinement of Rule Extraction**: The `_extract_rule_name` method could be further enhanced with more advanced NLP techniques (e.g., named entity recognition, semantic similarity) for more accurate rule identification from complex queries.
5.  **Advanced RAG Retrieval Strategies**: Implement re-ranking, hybrid search (keyword + vector), or graph-based retrieval for improved precision and recall.
6.  **Conversational Memory**: Develop more advanced conversational memory for the agent to maintain context over longer interactions.
7.  **Dynamic Context Window Management**: Automatically adjust context passed to the LLM based on query complexity and token limits.
8.  **Enhanced Output and Visualization**: Improve reporting with more structured JSON/XML outputs and interactive dashboards.
9.  **User Interface (UI)**: Develop a simple web-based UI for intuitive interaction.
10. **Automated Remediation**: Refine LLM's ability to provide actionable, context-aware remediation suggestions.
11. **Integration with Legal Databases**: Connect to external legal databases for automated rule updates and relevant precedents.

## Conclusion

The Policy Compliance Checker RAG System successfully establishes a foundational architecture for automated contract compliance analysis. Despite challenges with API quota limitations, the system demonstrates strong potential through its modular design, intelligent agent, and comprehensive reporting capabilities. Addressing the identified limitations, particularly regarding API stability and robust content parsing, will be crucial for evolving this project into a fully functional and reliable solution for legal tech applications.
