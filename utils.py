import re


def extract_clean_text(input_text):
    # Define the regex pattern
    pattern = r"(?<=\[\/INST\])(.*)"
    # Use re.search to find the match
    match = re.search(pattern, input_text)

    # Extract the text if a match is found
    if match:
        extracted_text = match.group(1)
        return extracted_text
