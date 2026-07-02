# HuggingFace Transformers

Notebooks covering the full HuggingFace ecosystem: pipeline API, tokenizer internals, dataset preparation, fine-tuning with the Trainer API, Retrieval-Augmented Generation with Llama 2, and LangChain integration with HuggingFace endpoints.

## Notebooks

| Notebook | Description |
|---|---|
| [01_transformers_pipeline_api](01_transformers_pipeline_api.ipynb) | Eight NLP tasks with the `pipeline()` API: sentiment analysis, zero-shot classification, text generation (DistilGPT2), fill-mask, NER with entity grouping, question answering, summarization, and French→English translation |
| [02_tokenizers_and_models](02_tokenizers_and_models.ipynb) | Pipeline internals: tokenization with AutoTokenizer (input IDs, attention masks, padding), forward pass with AutoModelForSequenceClassification, and softmax post-processing |
| [03_dataset_processing](03_dataset_processing.ipynb) | Dataset preparation for fine-tuning: MRPC loaded via Datasets, tokenized pairs with truncation, batch `.map()`, and dynamic batching with DataCollatorWithPadding |
| [04_finetuning_trainer_api](04_finetuning_trainer_api.ipynb) | Fine-tuned BERT on MRPC paraphrase detection with the HuggingFace Trainer API — TrainingArguments, custom `compute_metrics` (accuracy + F1), and validation tracking |
| [05_rag_llama2_langchain_chromadb](05_rag_llama2_langchain_chromadb.ipynb) | RAG system: Llama 2 7B (4-bit NF4 via BitsAndBytes), LangChain orchestration, ChromaDB vector store, Sentence Transformers embeddings (all-mpnet-base-v2), querying Biden's 2023 State of the Union |
| [06_langchain_huggingface_endpoints](06_langchain_huggingface_endpoints.ipynb) | LangChain with HuggingFace Inference Endpoints: PromptTemplate, LCEL pipe operator, and conversational memory patterns (ConversationBufferMemory, ConversationSummaryMemory) |

## Stack

- **HuggingFace:** Transformers, Datasets, Evaluate, Trainer API
- **Models:** BERT, DistilBERT, DistilGPT2, Llama 2 7b-chat-hf, Helsinki-NLP/opus-mt-fr-en
- **RAG stack:** LangChain · ChromaDB · Sentence Transformers · BitsAndBytes (4-bit NF4 quantization)
- **Framework:** PyTorch
