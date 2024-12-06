import json
import pandas as pd
from gradio_client import Client
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

# Set up the Gradio client
client = Client("https://leolm-leo-hessianai-13b-chat.hf.space/")

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

        # Send the request to the Hugging Face Space API
        response = client.predict(
            final_prompt,  # The formatted prompt as input text
            8192,            # Max tokens
            0.2,           # Temperature
            0,             # Top P
            0,             # Top K
            0,             # Repetition Penalty
            fn_index=3     # Use the function index for generating text
        )

        # Store the question and the response
        data.append({
            "Frage": query,
            "Antwort": response  # Access response content directly
        })

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)

    # Serialize the DataFrame and save it to a separate file for each prompt
    file_name = f"./data/llama-32-3B/{prompt_choice}.csv"
    df.to_csv(file_name, index=False)

    # Print a confirmation message for each file
    print(f"DataFrame for '{prompt_choice}' has been serialized and saved as '{file_name}'")
