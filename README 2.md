# RAG Project Virtual Environment Setup

This project contains a comprehensive RAG (Retrieval-Augmented Generation) implementation with various notebooks demonstrating different approaches.

## 🚀 Quick Start

### Option 1: Using the Activation Script (Recommended)
```bash
./activate_rag_env.sh
```

### Option 2: Manual Activation
```bash
cd "/Users/kekunkoya/Desktop/RAG Project"
source rag_env/bin/activate
```

## 📦 Installed Packages

The virtual environment includes all necessary packages for RAG development:

### Core RAG Libraries
- **OpenAI API** - For embeddings and chat completions
- **PyPDF2** - PDF text extraction
- **PyMuPDF (Fitz)** - Alternative PDF processing

### Machine Learning & Data Processing
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning utilities
- **FAISS** - Vector similarity search
- **Transformers** - Hugging Face transformers

### NLP & Text Processing
- **SpaCy** - Natural language processing
- **Tiktoken** - Token counting
- **Rank-BM25** - Traditional information retrieval

### Visualization & Development
- **Matplotlib** - Plotting
- **Jupyter** - Interactive notebooks
- **IPyWidgets** - Interactive widgets

## 🎯 Available Notebooks

1. **01_simple_ragKemi Query.ipynb** - Basic RAG implementation
2. **01_simple_ragKemiPEMA.ipynb** - RAG with PEMA dataset
3. **04_context_enriched_ragKemi.ipynb** - Context-enriched RAG
4. **07_query_transformKemi.ipynb** - Query transformation techniques
5. **12_adaptive_ragKemi.ipynb** - Adaptive RAG approaches
6. **15_multimodel_ragKemi.ipynb** - Multi-model RAG
7. **20_cragKemi.ipynb** - Contextual RAG
8. **best_rag_finder.ipynb** - RAG evaluation and comparison

## 🔧 Usage Instructions

### Starting Jupyter Notebook
```bash
# After activating the environment
jupyter notebook
```

### Running a Specific Notebook
```bash
# After activating the environment
jupyter notebook 01_simple_ragKemi\ Query.ipynb
```

### Deactivating the Environment
```bash
deactivate
```

## 🔑 Environment Variables

Make sure to set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project directory:
```
OPENAI_API_KEY=your-api-key-here
```

## 📁 Project Structure

```
RAG Project/
├── rag_env/                    # Virtual environment
├── PDFs/                       # PDF documents
├── *.ipynb                     # Jupyter notebooks
├── *.json                      # Dataset files
├── requirements.txt            # Python dependencies
├── activate_rag_env.sh         # Activation script
└── README.md                   # This file
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Make sure the virtual environment is activated
2. **API Key Issues**: Verify your OpenAI API key is set correctly
3. **PDF Processing**: Ensure PDF files are in the correct location

### Reinstalling Dependencies
```bash
pip install -r requirements.txt
```

## 📚 Next Steps

1. Start with `01_simple_ragKemi Query.ipynb` to understand basic RAG
2. Explore different RAG approaches in the other notebooks
3. Experiment with your own datasets and queries
4. Evaluate performance using the evaluation notebooks

## 🤝 Support

If you encounter any issues, check:
1. Virtual environment activation
2. API key configuration
3. File paths and dependencies
4. Python version compatibility (Python 3.13+ recommended) 