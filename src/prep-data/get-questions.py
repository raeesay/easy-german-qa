import os
import random
import json
import xml.etree.ElementTree as ET

def extract_qa_pairs_random_sample(input_dir, output_file, sample_size=50):
    # Storage for questions and answers
    qa_pairs = []

    # Get a list of all XML files in the directory
    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]

    # Randomly select a sample of up to 'sample_size' XML files
    sample_files = random.sample(xml_files, min(sample_size, len(xml_files)))

    for file_name in sample_files:
        file_path = os.path.join(input_dir, file_name)
        try:
            # Parse XML and extract questions and answers
            tree = ET.parse(file_path)
            root = tree.getroot()
            for qa_pair in root.findall(".//QAPair"):
                question_elem = qa_pair.find("Question")
                answer_elem = qa_pair.find("Answer")
                
                # Extract question and answer text if available
                question_text = question_elem.text.strip() if question_elem is not None else None
                answer_text = answer_elem.text.strip() if answer_elem is not None else None
                
                # Add question and answer to the list
                if question_text and answer_text:
                    qa_pairs.append({
                        "file": file_name,
                        "question": question_text,
                        "answer": answer_text
                    })
                
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    # Write questions and answers to JSON file
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(qa_pairs, json_file, ensure_ascii=False, indent=4)

    print(f"Extracted QA pairs saved to {output_file}")

# Usage example
input_dir = './data/MedQuAD/'
output_file = './src/prep-data/sampled_qa_pairs.json'
extract_qa_pairs_random_sample(input_dir, output_file)
