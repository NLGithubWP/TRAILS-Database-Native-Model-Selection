

import os
import random


def split_file_shuffle(input_file, num_segments):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    random.shuffle(lines)

    lines_per_segment = len(lines) // num_segments

    segments = [lines[i:i + lines_per_segment] for i in range(0, len(lines), lines_per_segment)]

    for i, segment in enumerate(segments):
        with open(f'valid_{i}.libsvm', 'w') as f:
            f.writelines(segment)


def split_file(input_file, num_segments, namespace):
    # Count the total number of rows in the input file
    with open(input_file, 'r') as f:
        num_rows = sum(1 for line in f)
        print(f"1. count rows = {num_rows}")

    # Calculate the number of rows per segment
    rows_per_segment = num_rows // num_segments
    if num_rows % num_segments != 0:
        rows_per_segment += 1

    # Create the output directory if it doesn't exist
    output_dir = 'output_'+namespace
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Open the input file for reading
    with open(input_file, 'r') as f:
        current_segment = 1
        rows_written = 0
        output_file = open(f'{output_dir}/{namespace}_segment_{current_segment}.libsvm', 'w')

        # Read the input file line by line
        for line in f:
            output_file.write(line)
            rows_written += 1

            # If the current segment is full, close the file and move on to the next segment
            if rows_written == rows_per_segment:
                output_file.close()
                current_segment += 1
                rows_written = 0
                output_file = open(f'{output_dir}/{namespace}_segment_{current_segment}.libsvm', 'w')

        # Close the final output file
        output_file.close()


if __name__ == '__main__':
    num_segments = 30
    # split the train
    input_file = '../exp_data/data/structure_data/frappe/train.libsvm'
    split_file(input_file, num_segments, "train")



