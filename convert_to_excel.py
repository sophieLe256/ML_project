import pandas as pd

num = [1, 2, 3] 

for i in num:
    txt_file = f"MissingData{i}.txt"  
    output_excel = f"output_MissingData{i}.xlsx"  \

    try:
        df = pd.read_csv(txt_file, delim_whitespace=True, header=None)
        df.to_excel(output_excel, index=False, header=False)

        print(f"File successfully converted to {output_excel}")
    except FileNotFoundError:
        print(f"File {txt_file} not found.")
    except Exception as e:
        print(f"An error occurred while processing {txt_file}: {e}")
