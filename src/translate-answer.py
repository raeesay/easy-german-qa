import os
import re
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI 
from langchain.schema import HumanMessage, SystemMessage

# Load prompts from JSON
with open("./src/prep-data/translation-prompts.json", "r") as file:
    prompts = json.load(file)

prompt_choices = ["crazy", "simple", "easy", "rules", "example-text", "three-shot-aligned", "three-shot-plain", "five-shot", "rules-example"]

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
    "mistral-large": ChatMistralAI(
        model="mistral-large-latest",  
        temperature=0.1,
        frequency_penalty=0.1
    ),
    "mistral-nemo": ChatMistralAI(
        model="open-mistral-nemo",  
        temperature=0.1,
        frequency_penalty=0.1
    ) 
}

# Define a function to extract the content from 'Antwort'
def extract_content(antwort):
    # Use regex to find the content between 'content=' and the next closing quote
    match = re.search(r"content='(.*?)'", antwort)
    if match:
        return match.group(1)
    return antwort  # Return original if no match is found

# Loop through each LLM
for model_name, llm in llms.items():
    print(f"Processing with model: {model_name}")

    if model_name == "gpt":
        df = pd.read_csv("./data/gpt/standard.csv")
    elif model_name == "gpt-mini":
        df = pd.read_csv("./data/gpt-mini/standard.csv")
    elif model_name == "llama-3B":
        df = pd.read_csv("./data/llama-3B/standard.csv")
    elif model_name == "llama-70B":
        df = pd.read_csv("./data/llama-70B/standard.csv")
    elif model_name == "mistral-large":
        df = pd.read_csv("./data/mistral-large/standard.csv")
    elif model_name == "mistral-nemo":
        df = pd.read_csv("./data/mistral-nemo/standard.csv")

    # Loop through each prompt
    for prompt_choice in prompt_choices:
        selected_prompt = prompts[prompt_choice]["prompt"]

        # Create an empty list to store translations for this prompt
        translations = []

        # Loop through each row in the DataFrame
        for index, row in df.iterrows():
            # Extract the content of the 'Antwort' column
            antwort_content = extract_content(row['Antwort'])

            # Create the conversation prompt for the LLM
            prompt = [
                SystemMessage(content=selected_prompt),
                HumanMessage(content=f"\n\nText: {antwort_content}. \nÜbersetzung: ")
            ]

            # Send the prompt to the LLM and get the translation
            response = llm.invoke(prompt)

            # Extract the translated content from the response
            translation = response

            # Append the translation to the list
            translations.append(translation)

        # Add the translations as a new column to the dataframe
        df['Übersetzung'] = translations

        # Optionally, save the updated dataframe to a new file
        file_name = f"./data/{model_name}/translated-answer/{prompt_choice}.csv"
        df.to_csv(file_name, index=False)

        # Print a confirmation message for each file
        print(f"DataFrame for '{model_name}' with prompt '{prompt_choice}' has been serialized and saved as '{file_name}'")
