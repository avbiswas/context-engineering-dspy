# Context Engineering Tutorial

This repo contains the source code from my Youtube course on Context Engineering with DSPy.

> **📺 Watch the Course for free**  
> **[Context Engineering - Complete 1h 20m Course](https://youtu.be/5Bym0ffALaU?si=gOLDiT-IVE7CxRwX)**  
> *Learn advanced prompt engineering techniques with hands-on examples*

## Support

If you find this content helpful, please consider supporting my work on Patreon. Your support helps me create more in-depth tutorials and content. My Patreon hosts all the code, projects, slides, write-ups I have ever made on my YouTube channel. 

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdown)

## Getting Started

### Prerequisites

-   **Python 3.10+** (required)
-   **`uv`** (recommended) or `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/avbiswas/context-engineering-dspy
    cd context-engineering-dspy/tutorial
    ```

2.  **Install dependencies:**
    ```bash
    # Using uv
    uv sync
    ```

3.  **Set up your API keys:**
    
    **Required API Keys:**
    - `OPENAI_API_KEY` - For OpenAI models
    - `GEMINI_API_KEY` - For Google Gemini models  
    - `TAVILY_API_KEY` - For web search functionality
    
    **Environment Management Options:**
    
    **Option 1: Using `direnv` (Recommended)**
    ```bash
    # Install direnv first, then create .envrc file
    echo "export OPENAI_API_KEY=your_key_here" >> .envrc
    echo "export GEMINI_API_KEY=your_key_here" >> .envrc
    echo "export TAVILY_API_KEY=your_key_here" >> .envrc
    direnv allow
    ```
    
    **Option 2: Using `.env` file with python-dotenv**
    ```bash
    # Create .env file
    touch .env
    ```
    Add your keys to `.env`:
    ```env
    OPENAI_API_KEY=your_key_here
    GEMINI_API_KEY=your_key_here
    TAVILY_API_KEY=your_key_here
    ```
    *Note: This requires adding `dotenv.load_dotenv()` to your Python scripts.*
    
    **Option 3: Global environment variables** *(Not recommended for security)*
    ```bash
    export OPENAI_API_KEY=your_key_here
    # Repeat for other keys...
    ```

4.  **Run the examples:**
    Navigate to any level directory and run the Python scripts:
    ```bash
    cd level2_multi_interaction
    uv run t1_sequential_flow.py
    ```

## File Descriptions

### Level 1: Atomic Prompts

-   `level1_atomic_prompts/level1.ipynb`: Introduces the basics of prompting and interacting with language models.

### Level 2: Multi-Interaction

-   `level2_multi_interaction/t1_sequence.py`: Basic sequence example.
-   `level2_multi_interaction/t1_sequential_flow.py`: Demonstrates a sequential flow of interactions with the language model.
-   `level2_multi_interaction/t2_iterative_refinement.py`: Shows how to iteratively refine the output from the model.
-   `level2_multi_interaction/t3_conditional_branch.py`: Illustrates how to use conditional logic to guide the conversation with the model.
-   `level2_multi_interaction/t3-multi_out.py`: Multiple output handling example.
-   `level2_multi_interaction/t3-multi_out_refine.py`: Refined multiple output handling.
-   `level2_multi_interaction/t4_reflection.py`: An example of how to make the model reflect on its own output.

### Level 3: Evaluation

-   `level3_evaluation/analysis.ipynb`: Analysis notebook for evaluation techniques.
-   `level3_evaluation/pairwise_elo.py`: Explains how to use the Elo rating system for pairwise comparison of model outputs.
-   `level3_evaluation/reflection.py`: Shows how to use reflection for evaluation.

### Level 4: Tools

You will need the TAVILY_API_KEY to run web search. You can sign up for a free account from their website.

-   `level4_tools/main.py`: Main tool usage examples.
-   `level4_tools/tool_calling_agent.py`: An example of a tool-calling agent.
-   `level4_tools/tools.py`: Tool definitions and implementations.
-   `level4_tools/idea_gen.py`: Idea generation tool example.
-   `level4_tools/joke_gen.py`: Joke generation tool example.

### Level 5: RAGs (Retrieval-Augmented Generation)

First, download this dataset:
https://www.kaggle.com/datasets/abhinavmoudgil95/short-jokes

Unzip inside `level5/data`
Next, prepare the embeddings:
```
cd level5_rags
uv run vector_embedding.py
```

This code looks for the file `level5_rags/data/shortjokes.csv`
This will create some files inside the `data/` directory. You should now be able to run scripts to play with retrieval.

**Core RAG Implementations:**
-   `level5_rags/basic_rag.py`: A basic RAG implementation.
-   `level5_rags/hyde.py`: An implementation of the HyDE (Hypothetical Document Embeddings) technique.
-   `level5_rags/annoy_rag.py`: RAG implementation using Annoy for vector similarity.

**Retrieval Components:**
-   `level5_rags/bm25_retriever.py`: BM25-based retrieval implementation.
-   `level5_rags/rank_fusion.py`: An example of fusing ranks from multiple retrievers.
-   `level5_rags/vector_embedding.py`: Vector embedding utilities.

**Tools & Applications:**
-   `level5_rags/main.py`: Main application with RAG-powered tools.
-   `level5_rags/tools.py`: Tool definitions for RAG applications.
-   `level5_rags/joke_gen.py`: Joke generation using RAG.
-   `level5_rags/idea_gen.py`: Idea generation using RAG.

**Utilities:**
-   `level5_rags/prepare_data.py`: Data preparation utilities for RAG systems.
-   `level5_rags/data/`: Directory containing data files for RAG examples.


## Quick Start Patterns

To run examples from each level:

```bash
# Level 2 - Multi-interaction examples
cd level2_multi_interaction
uv run t1_sequential_flow.py
uv run t2_iterative_refinement.py
```
