import pandas as pd

# List of indices for the files
num = [1, 2, 3]  # Assuming TrainData1.txt to TrainData6.txt

for i in num:
    # Construct file names dynamically
    txt_file = f"MissingData{i}.txt"  # Replace with your actual .txt file names
    output_excel = f"output_MissingData{i}.xlsx"  # Corresponding Excel output file

    try:
        # Read the text file with whitespace delimiter
        df = pd.read_csv(txt_file, delim_whitespace=True, header=None)

        # Save DataFrame to Excel
        df.to_excel(output_excel, index=False, header=False)

        print(f"File successfully converted to {output_excel}")
    except FileNotFoundError:
        print(f"File {txt_file} not found.")
    except Exception as e:
        print(f"An error occurred while processing {txt_file}: {e}")
