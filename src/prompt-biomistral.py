import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import HumanMessage, SystemMessage

# Load prompts from JSON
with open("./src/prompts.json", "r") as file:
    prompts = json.load(file)

# Load queries from JSON
with open("./src/queries.json", "r") as file:
    query_data = json.load(file)
    queries = query_data["queries"]

prompt_choices = ["example-text", "three-shot-aligned", "three-shot-plain", "five-shot", "rules-example"]

# Initialize vectorstore and embedding model
vectorstore = Chroma(persist_directory="./data/")
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# Set up the BioMistral model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B")

# Loop through each prompt
for prompt_choice in prompt_choices:
    selected_prompt = prompts[prompt_choice]["prompt"]

    # Initialize a list to store results for this prompt
    data = []

    for query in queries:
        # Embed the query
        query_embedding = embedding_model.embed_query(query)

        # Perform similarity search using the embedded query
        results = vectorstore.similarity_search_by_vector(query_embedding, k=5)  # k specifies top-k results

        # Combine the retrieved chunks as context
        context = "\n\n".join([result.page_content for result in results])

        # Format the prompt with context and query
        final_prompt = f"{selected_prompt}\n\nInformationen: {context}. \nFrage: {query}. \nAntwort: "

        # Encode the prompt and generate the response
        inputs = tokenizer(final_prompt, return_tensors="pt", max_length=2048, truncation=True)
        outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.1, top_k=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store the question and the response
        data.append({
            "Frage": query,
            "Antwort": response
        })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Serialize the DataFrame and save it to a separate file for each prompt
    file_name = f"./data/biomistral-7B/{prompt_choice}.csv"
    df.to_csv(file_name, index=False)

    # Print a confirmation message for each file
    print(f"DataFrame for '{prompt_choice}' has been serialized and saved as '{file_name}'")
