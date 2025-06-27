import sys

def resolve_git_conflicts(ipynb_path, output_path=None):
    if output_path is None:
        output_path = ipynb_path  # Overwrite in place if no output path is given

    with open(ipynb_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    in_conflict = False
    keep_lines = False

    for line in lines:
        if line.startswith("<<<<<<<"):
            in_conflict = True
            keep_lines = False
            continue
        elif line.startswith("======="):
            keep_lines = True  # Start keeping the bottom version
            continue
        elif line.startswith(">>>>>>>"):
            in_conflict = False
            keep_lines = False
            continue

        if not in_conflict or keep_lines:
            cleaned_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"Conflicts resolved. Clean file saved to: {output_path}")

# Example usage:
# resolve_git_conflicts("path/to/notebook.ipynb")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_ipynb_conflicts.py notebook.ipynb [output.ipynb]")
    else:
        resolve_git_conflicts(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
