import os
import re
import pandas as pd
import sacrebleu
from bert_score import score as bert_score
from bleurt import score

# Function to extract the value of "content=" from a string
def extract_content(answer):
    match = re.search(r"content='(.*?)'", answer)
    if match:
        return match.group(1)
    return answer  # Fallback: Return original text if "content=" is not found

# Define a function to process a single pair of reference and generated files
def process_files(reference_file, generated_file, output_folder, bleurt_checkpoint):
    # Load the data
    df_reference = pd.read_csv(reference_file)
    df_generated = pd.read_csv(generated_file)

    df_reference["Generated"] = df_generated["Antwort"]

    # Filter out rows where 'Antwort' or 'Generated' contain 'Ich weiß es nicht.'
    df = df_reference[~df_reference[['Antwort', 'Generated']].isin(["Ich weiß es nicht."]).any(axis=1)]
    df = df.reset_index(drop=True)

    # Function to compute BLEU score
    def compute_bleu_for_row(row):
        reference = row["Antwort"]  # Reference text
        hypothesis = row["Generated"]  # Generated text
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return bleu.score

    # Function to compute BERTScore
    def compute_bertscore_for_row(row):
        reference = row["Antwort"]
        hypothesis = row["Generated"]
        P, R, F1 = bert_score([hypothesis], [reference], lang="de")  # Compute BERTScore for German
        return P[0].item(), R[0].item(), F1[0].item()

    # Initialize BLEURT scorer
    scorer = score.BleurtScorer(bleurt_checkpoint)

    # Function to compute BLEURT score for a single row
    def compute_bleurt_for_row(row):
        reference = row["Antwort"]
        hypothesis = row["Generated"]
        scores = scorer.score(references=[reference], candidates=[hypothesis])
        return scores[0]

    # Apply the BLEU calculation to each row
    df["BLEU"] = df.apply(compute_bleu_for_row, axis=1)
    # Compute BERTScore and expand into three columns
    bert_scores = df.apply(compute_bertscore_for_row, axis=1)
    df["BERT_Precision"], df["BERT_Recall"], df["BERT_F1"] = zip(*bert_scores)
    # Apply BLEURT calculation to each row
    df["BLEURT"] = df.apply(compute_bleurt_for_row, axis=1)

    # Save the updated DataFrame to an Excel file
    os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists
    output_file = os.path.join(output_folder, os.path.basename(reference_file).replace('.csv', '_results.xlsx'))
    df.to_excel(output_file, engine='openpyxl')
    print(f"Processed {reference_file} -> Results saved to {output_file}")

bleurt_checkpoint = './BLEURT-20'  # Path to BLEURT checkpoint


# Define the files to process
llama_3_reference = './data/llama-3B/standard.csv'
llama_3_generated_files = './data/llama-3B/crazy.csv'
llama_3_output_folder = './results/on-the-fly/llama-3B/'  # Folder to save results

process_files(llama_3_reference, llama_3_generated_files, llama_3_output_folder, bleurt_checkpoint)

# Define the files to process
mixtral_reference = './data/mixtral-8x7B/standard.csv'
mixtral_generated_files = './data/mixtral-8x7B/crazy.csv'
mixtral_output_folder = './results/on-the-fly/mixtral-8x7B/'  # Folder to save results

process_files(mixtral_reference, mixtral_generated_files, mixtral_output_folder, bleurt_checkpoint)

