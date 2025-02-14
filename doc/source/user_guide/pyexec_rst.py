import pathlib
import re
import sys
from io import StringIO

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

    for line in lines:
        line = line.strip()
        if not line or line in (">>>", "..."):
            continue

        cleaned_lines.append("".join([line[4:]]))

    return "\n".join(cleaned_lines)


def extract_code_blocks(rst_content: str) -> list:
    """
    Extract Python code blocks from RST content using regex pattern matching.

    Args:
        rst_content (str): RST document content

    Returns:
        list: List of tuples containing (cleaned_code, has_parsed_literal, parsed_literal_position)
    """
    pattern = r"""
        \.\.?\s*code-block::\s*python\s*\n
        (?:\s*\n)?
        (\s+>>>.*?(?:\n\s+(?:>>>|\.\.\.).*?)*)
        (?=\n(?:\s*\n|\s+\n)|$)
        (?:\n+(?:[ \t]*\n)*\.\.?\s*parsed-literal::\s*\n\s*(.*?)(?=\n\s*\n|$))?
    """

    blocks = []
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
                last_line = code.split("\n")[-1]  # to print if it is a variable

                if last_line[0] == " ":
                    # Execute the entire block
                    exec(code, self.globals)  # noqa: S102

                else:  # separate the last line from the entire block
                    # Execute the entire block except the last line
                    exec(code[: -len(last_line)], self.globals)  # noqa: S102
                    # Try to print the last line output or execute it
                    try:
                        code_last_line = eval(compile(last_line, "<string>", "eval"), self.globals)
                        if code_last_line is not None:
                            print(repr(code_last_line))
                    except SyntaxError:
                        exec(last_line, self.globals)  # noqa: S102
                    except Exception as exception:
                        raise exception

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
        overwrite (bool): Overwrite the existing file or create a new file with updated content

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
            pattern = r"\.\.?\s*parsed-literal::\s*\n\s*(.*?)(?=\n\s*\n|$)"
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

        outputs = process_rst_file(str(file_path), overwrite)

        for i, (code, output_i) in enumerate(outputs, 1):
            print(f"Block {i} - Output:\n   ", output_i)

    else:
        raise FileNotFoundError(f"File not found at path: {file_path}")
