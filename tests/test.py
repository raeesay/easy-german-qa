import os

def rename_files_add_seven(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith('.xml'):
            # Split the file name into name and extension
            base_name, extension = os.path.splitext(file_name)
            
            # Check if the base name is numeric (as in '0000001')
            if base_name.isdigit():
                # Add "7" after the leading zeros
                new_base_name = base_name[:1] + '7' + base_name[1:]
                
                # Construct the new file name
                new_file_name = new_base_name + extension
                
                # Rename the file
                old_path = os.path.join(directory, file_name)
                new_path = os.path.join(directory, new_file_name)
                
                os.rename(old_path, new_path)
                print(f'Renamed {file_name} to {new_file_name}')

# Usage example
directory = './data/MedQuAD/7_SeniorHealth_QA'
rename_files_add_seven(directory)
