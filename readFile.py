import pandas as pd
import io

def readExcel(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    return df

    # Create a buffer
    buffer = io.StringIO()

    # Convert DataFrame to a dictionary of columns
    data_dict = {col: df[col].tolist() for col in df.columns}

    return data_dict

    # Write the data to the buffer
    for col, values in data_dict.items():
        buffer.write(f"{col}: {values}\n")

    # Get the buffer content as a string
    buffer_content = buffer.getvalue()

    # Close the buffer
    buffer.close()

    return buffer_content

# Example usage
#file_path = 'D://Users//PraveenS//Downloads//Chrome//Manufacturing+Downtime//Manufacturing_Line_Productivity.xlsx'
#buffer_content = readExcel(file_path)
#print(buffer_content)
