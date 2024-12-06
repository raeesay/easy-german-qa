import os
import pandas as pd
import re
import syllapy
import nltk
import spacy

# Load German language model
nlp = spacy.load("de_core_news_sm")

# Helper functions for text analysis
def sentence_count(text):
    sentences = re.split(r'[.!?]', text)
    return len([sentence for sentence in sentences if sentence.strip()])

def avg_sentence_length(text):
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
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
    words = nltk.word_tokenize(text)
    three_or_more_syllables = [word for word in words if syllapy.count(word) >= 3]
    return len(three_or_more_syllables) / len(words) * 100 if len(words) > 0 else 0

def percentage_words_more_than_six_letters(text):
    words = nltk.word_tokenize(text)
    long_words = [word for word in words if len(word) > 6]
    return len(long_words) / len(words) * 100 if len(words) > 0 else 0

def percentage_one_syllable_words(text):
    words = nltk.word_tokenize(text)
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

# Folder where your files are stored
folder_path = "./data/hand-aligned-corpus"  # Update this to the correct folder path

# List to store results
results = []

# Iterate through all .normal files in the directory
for file_name in os.listdir(folder_path):
    if file_name.endswith('.normal'):  # Look for .normal files
        base_name = file_name.split('.')[0]  # Get base name (e.g., '123' from '123.normal')
        
        normal_file = os.path.join(folder_path, file_name)
        simple_file = os.path.join(folder_path, f"{base_name}.simple")  # Corresponding .simple file
        
        if os.path.exists(simple_file):  # Check if the corresponding .simple file exists
            # Read reference and candidate texts
            with open(normal_file, 'r') as ref_file:
                reference_text = ref_file.read().strip()  # Read reference text
            with open(simple_file, 'r') as cand_file:
                candidate_text = cand_file.read().strip()  # Read candidate text

            
            # Compute readability and text complexity features for candidate text only
            sentence_count_can = sentence_count(candidate_text)
            avg_sentence_length_can = avg_sentence_length(candidate_text)
            avg_syllables_can = avg_syllables_per_word(candidate_text)
            flesch_reading_ease_can = flesch_reading_ease_de(candidate_text)
            flesch_kincaid_grade_can = flesch_kincaid_grade_level_de(candidate_text)
            wstf1_can = wstf1(candidate_text)
            wstf2_can = wstf2(candidate_text)
            wstf3_can = wstf3(candidate_text)
            wstf4_can = wstf4(candidate_text)
            
            # Store the results
            results.append({
               
                # Candidate Text Statistics (only for candidate text)
                "Sentence Count (Candidate)": sentence_count_can,
                "Avg Sentence Length (Candidate)": avg_sentence_length_can,
                "Avg Syllables per Word (Candidate)": avg_syllables_can,
                "Flesch Reading Ease (Candidate)": flesch_reading_ease_can,
                "Flesch Kincaid Grade Level (Candidate)": flesch_kincaid_grade_can,
                "WSTF1 (Candidate)": wstf1_can,
                "WSTF2 (Candidate)": wstf2_can,
                "WSTF3 (Candidate)": wstf3_can,
                "WSTF4 (Candidate)": wstf4_can
            })

# Convert the results to a DataFrame
df = pd.DataFrame(results)
print(df)
df.to_csv("benchmarks.csv")

