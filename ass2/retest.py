import re
digit_patterns = [re.compile('[0-9]+')]

def replace_digits(pair):
    for pattern in digit_patterns:
        if pattern.match(pair[0]):
            pair[0] = re.sub('[0-9]', 'DG', pair[0])
    return pair

def process_line_to_lower_test(line):
    return [replace_digits([tok.lower(), tag]) for tok, tag in [[pair, "NNP"] for pair in line.split()]]


def read_file_to_lower_test(file):
    with open(file, 'r') as file:
        data = file.read().replace('\n\n', '_')
        return [process_line_to_lower_test(line) for line in data.split("_") if not len(line) == 0]

print(read_file_to_lower_test("pos/test"))