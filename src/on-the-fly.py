import os
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI 
from langchain.schema import HumanMessage, SystemMessage

# Load the JSON data containing files, questions, and answers
with open("./src/prep-data/german-samples.json", "r") as file:
    query_data = json.load(file)

# Extract questions from JSON data
queries = [item["question"] for item in query_data]

# Load prompts from JSON
with open("./src/prep-data/prompts.json", "r") as file:
    prompts = json.load(file)

prompt_choices = ["standard", "crazy", "simple", "easy", "rules", "example-text", "three-shot-aligned", "three-shot-plain", "five-shot", "rules-example"]

# Initialize vectorstore and embedding model
vectorstore = Chroma(persist_directory="./data/")
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# LLM Initialization for each model
os.environ["OPENAI_API_KEY"] = "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"
os.environ["MISTRAL_API_KEY"] = "vdZSjaKEjVL8eiKk3NTwrINfQY7hN7t7"
HUGGINGFACEHUB_API_TOKEN = "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"

llms = {
    "gpt": ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        frequency_penalty=0.1
    ),
    "gpt-mini": ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        frequency_penalty=0.1
    ),
    "llama-3B": HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        temperature=0.1,
        frequency_penalty=0.1,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        timeout=300
    ),
    "llama-70B": HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-70B-Instruct",
        temperature=0.1,
        frequency_penalty=0.1,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        timeout=300
    ),
    "mistral-nemo": HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-Nemo-Instruct-2407",
        temperature=0.1,
        frequency_penalty=0.1,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        timeout=300
    ),
    "mixtral-8x7B": HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.1,
        frequency_penalty=0.1,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        timeout=300
    )
}

# Loop through each model
for model_name, llm in llms.items():
    print(f"Processing with model: {model_name}")

    # Loop through each prompt
    for prompt_choice in prompt_choices:
        selected_prompt = prompts[prompt_choice]["prompt"]

        # Initialize a list to store results for this prompt and model
        data = []

        for query in queries:
            # Embed the query
            query_embedding = embedding_model.embed_query(query)

            # Perform similarity search using the embedded query
            results = vectorstore.similarity_search_by_vector(query_embedding, k=5)

            # Combine the retrieved chunks as context
            context = "\n\n".join([result.page_content for result in results])

            # Format the prompt with context and query
            final_prompt = [
                SystemMessage(content=f"{selected_prompt} \n\nInformationen: {context}."),
                HumanMessage(content=f"\n\nFrage: {query}. \nAntwort: ")
            ]

            # Use other models like GPT, Llama, etc.
            response = llm.invoke(final_prompt)

            # Store the question and the response
            data.append({
                "Frage": query,
                "Antwort": response
            })

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        # Serialize the DataFrame and save it to a separate file for each prompt and model
        file_name = f"./data/{model_name}/{prompt_choice}.csv"
        df.to_csv(file_name, index=False)

        # Print a confirmation message for each file
        print(f"DataFrame for '{model_name}' with prompt '{prompt_choice}' has been serialized and saved as '{file_name}'")
