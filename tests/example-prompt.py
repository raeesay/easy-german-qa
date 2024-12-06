import os
import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

vectorstore = Chroma(persist_directory="./data/")
# Setting LLM model
os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# List of queries to embed and process
queries = ["Warum habe ich Zucker im Urin?", "Was passiert wenn ich kalt dusche?", "Was ist eine Blume?"]

# Initialize an empty list to store results
data = []

for query in queries:
    # Embed the query
    query_embedding = embedding_model.embed_query(query)

    # Perform similarity search using the embedded query
    results = vectorstore.similarity_search_by_vector(query_embedding, k=5)  # k specifies top-k results

    # Combine the retrieved chunks as context
    context = "\n\n".join([result.page_content for result in results])

    example1_prompt = f'''Dir wird eine Frage gestellt und Du erhälst die relevanten Informationen. Deine Aufgabe besteht darin, die Frage mit den bereitgestellten Informationen in leichter Sprache wie in dem Beispieltext zu beantworten. Wenn Du die Frage mit den gegebenen Informationen nicht beantworten kannst dann sag 'Ich weiß es nicht.'.
    Beispieltext:
        Mit unseren Texten in Einfacher Sprache wollen wir vielen Menschen diese Informationen zugänglich machen.
        Auf unserer Internetseite sollen sich alle Leser zum Thema Gesundheit informieren können."

        **Worum geht es in den Texten?**
        Auf der Internetseite der Apotheken Umschau gibt es sehr viele Informationen zum Thema Gesundheit.
        Zur Zeit gibt es etwa 130 Texte in Einfacher Sprache.
        Diese Texte informieren über ganz unterschiedliche Themen.

        Es gibt Texte zu vielen verschiedenen Krankheiten, zum Beispiel:
        - Adipositas
        - Gürtelrose
        - Weitsichtigkeit

        Es gibt Texte zu vielen Medikamenten und Heilpflanzen, zum Beispiel:
        - Ibuprofen
        - Arnika
        - Kurkuma

        Wie enstehen die Texte?
        Ein Übersetzer der Forschungsstelle Leichte Sprache übersetzt den Text.
        Dann prüfen zwei andere Übersetzer:
        Ist die Übersetzung verständlich geschrieben?
        Dann prüfen medizinische Fachleute vom Wort & Bild Verlag:
        Sind die Informationen in der Übersetzung richtig?
        Der Text ist sprachlich leicht verständlich geschrieben?
        Und die Informationen im Text sind richtig und vollständig?
        Erst dann kommen die Texte auf die Internetseite.
        \n\nInformationen: {context}\n\nFrage: {query}'''


    example2_message = [
        SystemMessage(content='''Dir wird eine Frage gestellt und Du erhälst die relevanten Informationen. Deine Aufgabe besteht darin, die Frage mit den bereitgestellten Informationen in leichter Sprache wie in dem Beispieltext zu beantworten. Wenn Du die Frage mit den gegebenen Informationen nicht beantworten kannst dann sag 'Ich weiß es nicht.'.
        Beispieltext:
        Das Netzwerk setzt sich für Leichte Sprache ein:
        - Im Alltag
        - In der Politik
        - In den Ämtern

        Das Netzwerk besteht seit dem Jahr 2006.
        Das Netzwerk ist seit dem Jahr 2013 ein Verein:
        Netzwerk Leichte Sprache e. V.

        Im Verein arbeiten viele Menschen zusammen.
        Zum Beispiel:
        - Das Netzwerk verbreitet die Leichte Sprache.
        - Das Netzwerk stärkt Menschen mit Lern-Schwierigkeiten.
        - Das Netzwerk möchte ein Recht auf Leichte Sprache für alle.

        Der Verein Netzwerk Leichte Sprache e. V. ist gemein-nützig.
        Das bedeutet:
        Die Arbeit vom Verein ist für alle Menschen.
        Der Verein ist wichtig für die Gesellschaft.

        ## Geschichte der Leichten Sprache
        Menschen mit Lern-Schwierigkeiten haben die Leichte Sprache mitentwickelt.
        Die Geschichte von der Leichten Sprache beginnt in Amerika.
        Das war vor 50 Jahren.
        Menschen mit Lern-Schwierigkeiten waren benachteiligt.
        Sie wollten sich das nicht mehr gefallen lassen.
        Sie haben für ihre Rechte gekämpft.
        Sie wollten zum Beispiel:
        - Gleich-Berechtigung für alle.
        - Mehr Selbst-Bestimmung für alle.

        20 Jahre später ist die Leichte Sprache nach Deutschland gekommen.
        Im Jahr 2001 wurde der Verein gegründet:
        **Mensch zuerst – Netzwerk People First Deutschland.**
        Der Verein hat viel für die Leichte Sprache gemacht.
        Der Verein hat Unterschriften für ein Recht auf Leichte Sprache gesammelt.
        Der Verein hat das erste Wörter-Buch für Leichte Sprache geschrieben.
        Das Netzwerk Leichte Sprache besteht seit 2006.
        Seit 2013 heißt es Netzwerk Leichte Sprache e. V.
        In den letzten Jahren ist viel passiert.
        Leichte Sprache wird immer bekannter.
        Immer mehr Menschen wollen Leichte Sprache.
        Die Leichte Sprache entwickelt sich immer weiter.
        '''),
        HumanMessage(content=f"Informationen: {context}\n\nFrage: {query}")
    ]

    # Generate responses for all message types
    example1_response = llm.invoke(example1_prompt)
    example2_response = llm.invoke(example2_prompt)

    # Store the results in the list as a dictionary
    data.append({
        "Frage": query,
        "Apotheken Umschau Beispiel": example1_response,
        "Netzwerk Leichte Sprache Beispiel": example2_response
    })

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame
print(df)