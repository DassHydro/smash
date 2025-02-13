import re
from io import StringIO
import sys
import pathlib
import pandas as pd


def clean_code_block(code_block: str) -> str:
    """
    Clean Python code blocks by removing REPL prompts and handling multiline statements.

    Args:
        code_block (str): Raw code block containing >>> and ... prompts

    Returns:
        str: Cleaned code with prompts removed and proper formatting
    """
    lines = code_block.split("\n")
    cleaned_lines = []
    current_statement = []
    in_function = False

    for line in lines:
        line = line.strip()
        if not line or line in (">>>", "..."):
            continue

        # Handle function definitions
        if line.startswith(">>> def "):
            if current_statement:
                cleaned_lines.append("".join(current_statement))
                current_statement = []
            in_function = True
            current_statement.append(line[4:])
            continue

        # Handle function body or continuation lines
        if line.startswith("... "):
            current_statement.append("\n" + line[4:] if in_function else line[4:])
        elif line.startswith(">>> "):
            if current_statement:
                cleaned_lines.append("".join(current_statement))
            current_statement = [line[4:]]
            in_function = False

    # Add any remaining statement
    if current_statement:
        cleaned_lines.append("".join(current_statement))

    return "\n".join(cleaned_lines)


def extract_code_blocks(rst_content: str) -> list:
    """
    Extract Python code blocks from RST content using regex pattern matching.

    Args:
        rst_content (str): RST document content

    Returns:
        list: List of tuples containing (cleaned_code, has_parsed_literal, parsed_literal_position)
    """
    blocks = []
    pattern = r"""
        \.\.?\s*code-block::\s*python\s*\n
        (?:\s*\n)?
        (\s+>>>.*?(?:\n\s+(?:>>>|\.\.\.).*?)*)
        (?=\n(?:\s*\n|\s+\n)|$)
        (?:\n+(?:[ \t]*\n)*\.\.?\s*parsed-literal::\s*\n\s*(.*?)(?=\n\s*\w|$))?
    """

    for match in re.finditer(pattern, rst_content, re.VERBOSE | re.DOTALL):
        code_block = match.group(1)
        has_parsed_literal = match.group(2) is not None
        parsed_literal_pos = match.end(1) if has_parsed_literal else None

        cleaned_code = clean_code_block(code_block)
        if cleaned_code:
            blocks.append((cleaned_code, has_parsed_literal, parsed_literal_pos))

    return blocks


class CodeExecutor:
    def __init__(self):
        self.globals = {}

    def execute_block(self, code: str) -> str:
        with StringIO() as redirected_output:
            old_stdout = sys.stdout
            sys.stdout = redirected_output

            try:
                if "def " in code:
                    # Execute the entire block at once for function definitions
                    try:
                        exec(code, self.globals)
                    except Exception as e:
                        print(f"Error: {str(e)}")
                else:
                    # Handle line by line for other code
                    for line in code.strip().split("\n"):
                        try:
                            result = eval(compile(line, "<string>", "eval"), self.globals)
                            if result is not None:
                                print(result)
                        except SyntaxError:
                            exec(line, self.globals)
                        except Exception as e:
                            print(f"Error: {str(e)}")

                output = redirected_output.getvalue().strip()
                return "\n".join(
                    [" " * 4 + line if i > 0 else line for i, line in enumerate(output.split("\n"))]
                    if output
                    else []
                )
            finally:
                sys.stdout = old_stdout


def process_rst_file(file_path: str, overwrite: bool) -> list:
    """
    Process an RST file by executing code blocks and updating parsed-literal sections.

    Args:
        file_path (str): Path to the RST file

    Returns:
        list: List of tuples containing (code, output) for each block
    """
    with open(file_path, "r") as file:
        content = file.read()

    blocks = extract_code_blocks(content)
    executor = CodeExecutor()

    # Execute all blocks to build state
    block_outputs = [executor.execute_block(code) for code, _, _ in blocks]

    # Update parsed-literal sections
    modified_content = list(content)
    outputs = []

    for (code, has_parsed_literal, parsed_literal_pos), output in zip(
        reversed(blocks), reversed(block_outputs)
    ):
        outputs.insert(0, (code, output))

        if has_parsed_literal:
            pattern = r"\.\.?\s*parsed-literal::\s*\n\s*(.*?)(?=\n\s*\w|$)"
            match = re.search(pattern, content[parsed_literal_pos:], re.DOTALL)

            if match:
                start = parsed_literal_pos + match.start(1)
                end = parsed_literal_pos + match.end(1)
                modified_content[start:end] = list(output)

    if not overwrite:
        file_path = file_path.replace(".rst", "_updated.rst")

    # Save updated content
    with open(file_path, "w") as file:
        file.write("".join(modified_content))

    return outputs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise IndexError("No file path provided to generate rst file")

    file_path = pathlib.Path(sys.argv[1])

    file_path = file_path.with_suffix(".rst")

    if file_path.exists():
        df = pd.read_csv("files_with_code_block.csv")
        if str(file_path) not in df["file_path"].to_list():
            path_input = input("Add file path to files_with_code_block.csv ([y]/n) ? ")

            if path_input.lower() in ["", "y", "yes"]:
                df = pd.concat([df, pd.DataFrame([{"file_path": str(file_path)}])], ignore_index=True)
                df.to_csv("files_with_code_block.csv", index=False)

        exists_input = input("Overwrite existing file ([y]/n) ? ")

        if exists_input.lower() in ["", "y", "yes"]:
            overwrite = True
        else:
            overwrite = False

        results = process_rst_file(str(file_path), overwrite)

        for i, (code, output) in enumerate(results, 1):
            print(f"\nBlock {i} - Output:\n   ", output)

    else:
        raise FileNotFoundError(f"File not found at path: {file_path}")
