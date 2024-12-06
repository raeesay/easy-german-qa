import os
import re
import spacy
import nltk
import string
import syllapy
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load German language model
nlp = spacy.load("de_core_news_sm")

# Helper functions
def sentence_count(text):
    sentences = re.split(r'[.!?]', text)
    return len([sentence for sentence in sentences if sentence.strip()])

def avg_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    words = word_tokenize(text)
    
    # Filter out punctuation
    words = [word for word in words if word not in string.punctuation]
    
    return len(words) / len(sentences) if len(sentences) > 0 else 0

def avg_syllables_per_word(text):
    # Process text with spaCy
    doc = nlp(text)
    
    # Tokenize and keep only words (exclude punctuations and spaces)
    words = [token.text for token in doc if token.is_alpha]
    
    # Simplified syllable counting for German words
    def count_syllables(word):
        return len(re.findall(r"[aeiouyäöü]+", word.lower()))  # Matches vowel groups
    
    # Calculate syllables per word
    syllables_per_word = [count_syllables(word) for word in words]
    
    # Compute average syllables per word
    return sum(syllables_per_word) / len(words) if len(words) > 0 else 0

def flesch_reading_ease_de(text):
    asl = avg_sentence_length(text)
    asw = avg_syllables_per_word(text)
    return 180 - asl - (58.5 * asw)

def flesch_kincaid_grade_level_de(text):
    asl = avg_sentence_length(text)
    asw = avg_syllables_per_word(text)
    return (0.39 * asl) + (11.8 * asw) - 15.59

def percentage_three_or_more_syllables(text):
    words = word_tokenize(text)
    three_or_more_syllables = [word for word in words if syllapy.count(word) >= 3]
    return len(three_or_more_syllables) / len(words) * 100 if len(words) > 0 else 0

def percentage_words_more_than_six_letters(text):
    words = word_tokenize(text)
    long_words = [word for word in words if len(word) > 6]
    return len(long_words) / len(words) * 100 if len(words) > 0 else 0

def percentage_one_syllable_words(text):
    words = word_tokenize(text)
    one_syllable_words = [word for word in words if syllapy.count(word) == 1]
    return len(one_syllable_words) / len(words) * 100 if len(words) > 0 else 0

def wstf1(text):
    ms = percentage_three_or_more_syllables(text)
    sl = avg_sentence_length(text)
    iw = percentage_words_more_than_six_letters(text)
    es = percentage_one_syllable_words(text)
    return 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875

def wstf2(text):
    ms = percentage_three_or_more_syllables(text)
    sl = avg_sentence_length(text)
    iw = percentage_words_more_than_six_letters(text)
    es = percentage_one_syllable_words(text)
    return 0.2007 * ms + 0.1682 * sl + 0.1373 * iw - 2.779

def wstf3(text):
    ms = percentage_three_or_more_syllables(text)
    sl = avg_sentence_length(text)
    iw = percentage_words_more_than_six_letters(text)
    es = percentage_one_syllable_words(text)
    return 0.2963 * ms + 0.1905 * sl - 1.1144

def wstf4(text):
    ms = percentage_three_or_more_syllables(text)
    sl = avg_sentence_length(text)
    iw = percentage_words_more_than_six_letters(text)
    es = percentage_one_syllable_words(text)
    return 0.2744 * ms + 0.2656 * sl - 1.693

# Function to count how many answers contain "Ich weiß es nicht."
def count_idk(df):
    return df['Antwort'].str.contains("Ich weiß es nicht.", case=False, na=False).sum()

# Update function to handle single/double quotes and apply it
def extract_content(answer):
    match = re.search(r"content=['\"](.*?)['\"]", answer)  # Match single or double quotes
    if match:
        return match.group(1)
    return answer

# Function to process a dataset and calculate metrics
def process_dataset(df):
    # Count the "Ich weiß es nicht." occurrences before removal
    idk_count_before = count_idk(df)

    # Remove rows with "Ich weiß es nicht." from the dataset
    df_cleaned = df[~df['Antwort'].str.contains("Ich weiß es nicht.", case=False, na=False)]

    # Count the remaining rows after removal
    idk_count_after = len(df) - idk_count_before
    total_answers_after = len(df_cleaned)
    total_answers_before = len(df)

    # Now calculate metrics only on the cleaned dataset (after removal)
    metrics = []
    for _, row in df_cleaned.iterrows():
        text = row['Antwort']
        metrics.append({
            "sentence_count": sentence_count(text),
            "avg_sentence_length": avg_sentence_length(text),
            "avg_syllables_per_word": avg_syllables_per_word(text),
            "flesch_reading_ease_de": flesch_reading_ease_de(text),
            "flesch_kincaid_grade_level_de": flesch_kincaid_grade_level_de(text),
            "wstf1": wstf1(text),
            "wstf2": wstf2(text),
            "wstf3": wstf3(text),
            "wstf4": wstf4(text),
            "idk": idk_count_before,
            "answers": total_answers_before,
            "after cleaning": total_answers_after
        })

    return pd.DataFrame(metrics)

# Function to calculate the required statistics for each metric
def compute_statistics(df):
    stats = {}
    for column in df.columns:
        stats[f"mean_{column}"] = df[column].mean()
        stats[f"median_{column}"] = df[column].median()
        stats[f"std_{column}"] = df[column].std()
    return stats

# Define models and dataset categories
models = ["gpt", "gpt-mini", "llama-70B", "llama-3B", "mixtral-8x7B"]
datasets = ["crazy", "easy", "example-text", "five-shot", "rules-example", 
            "rules", "simple", "standard", "three-shot-aligned", "three-shot-plain"]

# Create an empty DataFrame to store individual results
columns = ["sentence_count", "avg_sentence_length", "avg_syllables_per_word", 
           "flesch_reading_ease_de", "flesch_kincaid_grade_level_de", 
           "wstf1", "wstf2", "wstf3", "wstf4"]

# Iterate over datasets and models
for file in datasets:
    dataset_summary = []  # To hold summary data for this dataset
    for model in models:
        filepath = f"./data/{model}/{file}.csv"
        if os.path.exists(filepath):  # Ensure file exists
            df = pd.read_csv(filepath)
            if model in ["gpt", "gpt-mini"]:
                df['Antwort'] = df['Antwort'].apply(extract_content)
            df_processed = process_dataset(df)
            stats = compute_statistics(df_processed)

            # Add stats to the dataset summary
            dataset_summary.append({
                "model": model,
                **stats,
                "idk_count": df_processed["idk"].iloc[0],
                "total_answers_before": df_processed["answers"].iloc[0],
                "total_answers_after": df_processed["after cleaning"].iloc[0]
            })

    # Create a summary DataFrame for this dataset
    dataset_summary_df = pd.DataFrame(dataset_summary)

    # Save the dataset summary to a file (e.g., crazy-summary.df, easy-summary.df)
    summary_filepath = f'./results/on-the-fly/{file}.xlsx'
    dataset_summary_df.to_excel(summary_filepath, engine='openpyxl')

    print(f"Summary statistics for {file} have been calculated and saved.")
