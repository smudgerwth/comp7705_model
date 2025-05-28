import csv
import argparse

def extract_rows(input_file, output_file, num_rows=1000, header=True, remaining=False):
    """
    Extract rows from a CSV file. Can extract first N rows or remaining rows after N.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        num_rows (int): Number of rows to extract/skip (default: 1000)
        header (bool): Whether to keep the header row
        remaining (bool): If True, extract rows AFTER the first N rows
    """
    with open(input_file, 'r', newline='') as infile, \
         open(output_file, 'w', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        if header:
            header_row = next(reader)
            writer.writerow(header_row)

        rows_written = 0
        
        if remaining:
            # Skip specified number of rows
            for _ in range(num_rows):
                try:
                    next(reader)
                except StopIteration:
                    break  # No more rows to skip
            
            # Write all remaining rows
            for row in reader:
                writer.writerow(row)
                rows_written += 1
        else:
            # Write first N rows
            for row in reader:
                if rows_written >= num_rows:
                    break
                writer.writerow(row)
                rows_written += 1
    
    action = "remaining" if remaining else "first"
    print(f"Extracted {action} {rows_written} rows to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract rows from a CSV")
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument("-n", "--num_rows", type=int, default=1000,
                       help="Number of rows to extract/skip (default: 1000)")
    parser.add_argument("--no-header", action="store_false", dest="header",
                       help="Input file has no header row")
    parser.add_argument("--remaining", action="store_true",
                       help="Extract rows AFTER the first N rows instead of the first N rows")
    
    args = parser.parse_args()
    extract_rows(args.input, args.output, args.num_rows, args.header, args.remaining)
