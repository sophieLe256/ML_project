import pandas as pd

# List of indices for the files
num = [1, 2, 3]  # Assuming output_MissingData1.xlsx to output_MissingData3.xlsx

for i in num:
    # Construct file names dynamically
    input_excel = f"./MissingValues/MissingData{i}.xlsx"
    txt_file = f"MissingData{i}.txt"

    try:
        # Read the Excel file
        df = pd.read_excel(input_excel, header=None)

        # Save DataFrame to a text file with whitespace as delimiter
        df.to_csv(txt_file, sep=' ', index=False, header=False)

        print(f"File successfully converted to {txt_file}")
    except FileNotFoundError:
        print(f"File {input_excel} not found.")
    except Exception as e:
        print(f"An error occurred while processing {input_excel}: {e}")
