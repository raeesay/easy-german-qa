# Large Language Models for Question-Answering in Easy German - Prompting for Text Simplication
## Master's Thesis 2024 by Raeesa Yousaf



## Overview
This project explores the application of Large Language Models (LLMs) to simplify text and improve accessibility for non-native German speakers and individuals with cognitive disabilities. The study focuses on health-related questions, transforming complex medical information into Easy German through a domain-agnostic Question-Answering (QA) framework.  

### Key Features
- **Simplification Rules**: Explicit strategies to ensure outputs are clear, accurate, and accessible.
- **Prompting Strategies**: Comparison of three approaches:
  1. **On-the-Fly**: Direct generation.
  2. **Translated-Answer**: Two-step process involving intermediate translations.
  3. **Translated-Context**: Context-first translation before answering.
- **Model Comparison**: Evaluation of GPT-4o, Llama, and Mixtral for their effectiveness in producing factually accurate and coherent Easy German responses.

### Key Findings
- **GPT-4o** consistently outperformed open-source models (Llama and Mixtral) in producing accessible and accurate Easy German responses.
- Open-source models often struggled with coherence, generating repetitive or disjointed outputs.
- The **Translated-Answer** strategy enhanced readability and focus, demonstrating improvements over direct generation.

### Limitations
While GPT-4o demonstrated significant strengths, its closed-source nature limits accessibility and adoption. However, LLMs still offer a scalable and efficient alternative to training domain-specific models from scratch.

---
## Project Structure

The repository is organized as follows:
```
.
├── data/                 # Contains generated responses from LLMs
├── results/              # Contains evaluation results
├── tests/                # Contains initial testing (not relevant to the project)
├── src/                  # Contains code and resources for the project
│   ├── prep-data/        # Code for data preparation: chunking, embedding, and storing articles
│   │   ├── articles/     # JSON files of Apotheken Umschau articles
│   │   ├── prompts/      # JSON files of questions and prompts
│   ├── evaluation/       # Code for evaluating generated responses
│   ├── prompt_models.py  # Code for prompting various models

```

## How to Use

1. **Setup**:
   - Ensure all dependencies are installed (list them in `requirements.txt` if needed).

2. **Data Preparation**:
   - Use the scripts in `src/prep-data/` to chunk and embed articles, and store them as required.

3. **Generate Responses**:
   - Use `src/on-the-fly.py`, `src/translate-answer.py` and `src/translate-context.py` to generate responses from LLMs.

4. **Evaluate Responses**:
   - Run the evaluation scripts in `src/evaluation/` to assess readability, semantic alignment, and factual accuracy.

5. **View Results**:
   - Check the `results/` folder for evaluation outputs and analysis.