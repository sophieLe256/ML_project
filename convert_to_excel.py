import pandas as pd

# List of indices for the files
classify = [1, 2, 3, 4, 5]
missing = [1, 2, 3]
delimiter = r'[,\s]+'

for i in classify:
    # Construct file names dynamically
    txt_file_train = f"./txt_data/TrainData{i}.txt"
    txt_file_test = f"./txt_data/TestData{i}.txt" 
    output_excel_train = f"./Excel/output_TrainData{i}.xlsx"
    output_excel_test = f"./Excel/output_TestData{i}.xlsx"

    try:
        # Read the text file with whitespace delimiter
        df_train = pd.read_csv(txt_file_train, sep=delimiter, header=None)
        df_test = pd.read_csv(txt_file_test, sep=delimiter, header=None)

        # Save DataFrame to Excel
        df_train.to_excel(output_excel_train, index=False, header=False)
        df_test.to_excel(output_excel_test, index=False, header=False)

        print(f"Files successfully converted to {output_excel_train} and {output_excel_test}")
    except FileNotFoundError:
        print(f"Files for {i} not found.")
    except Exception as e:
        print(f"An error occurred while processing data{i}: {e}")

for i in missing:
    # Construct file names dynamically
    txt_file = f"./txt_data/MissingData{i}.txt"
    output_excel = f"./Excel/output_MissingData{i}.xlsx"

    try:
        # Read the text file with whitespace delimiter
        df = pd.read_csv(txt_file, sep=delimiter, header=None)

        # Save DataFrame to Excel
        df.to_excel(output_excel, index=False, header=False)

        print(f"Files successfully converted to {output_excel}")
    except FileNotFoundError:
        print(f"File {txt_file} not found.")
    except Exception as e:
        print(f"An error occurred while processing {output_excel}: {e}")