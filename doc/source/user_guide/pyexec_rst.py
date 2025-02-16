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
        list: List of tuples containing (cleaned_code, has_output_directive, output_directive_position)
    """
    pattern = r"""
        # match optional whitespace, '.. code-block:: python', and a newline
        \s*\.\.?\s*code-block::\s*python\s*\n
        # optionally match a blank line or a line with only whitespace
        (?:\s*\n)?
        # capture Python interactive session lines starting with '>>>'
        (\s+>>>.*?(?:\n\s+(?:>>>|\.\.\.).*?)*)
        # assert that the block is followed by a blank line, a line with whitespace, or the end of the string
        (?=\n(?:\s*\n|\s+\n)|$)
        # optionally match '.. code-block:: output' and capture the output block
        (?:\n+(?:[ \t]*\n)*\s*\.\.?\s*code-block::\s*output\s*\n\s*(.*?)(?=\n\s*\n|$))?
    """

    blocks = []
    for match in re.finditer(pattern, rst_content, re.VERBOSE | re.DOTALL):
        code_block = match.group(1)
        has_output_directive = match.group(2) is not None
        output_directive_pos = match.end(1) if has_output_directive else None

        cleaned_code = clean_code_block(code_block)
        if cleaned_code:
            blocks.append((cleaned_code, has_output_directive, output_directive_pos))

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

                if last_line.startswith((" ", "\t", ")", "]", "}")):
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

                output = redirected_output.getvalue().rstrip()
                return "\n".join(output.split("\n") if output else [])

            finally:
                sys.stdout = old_stdout


def process_rst_file(file_path: str, overwrite: bool) -> list:
    """
    Process an RST file by executing code blocks and updating output blocks.

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

    # Update output blocks
    modified_content = list(content)
    outputs = []

    for (_, has_output_directive, output_directive_pos), output in zip(
        reversed(blocks), reversed(block_outputs)
    ):
        if has_output_directive:
            # Match '.. code-block:: output', optional blank line, capture lines,
            # ensure followed by blank line/end
            pattern = r"\s*\.\.?\s*code-block::\s*output\s*\n\s*\n(\s*.*?)(?=\n\s*\n|$)"
            match = re.search(pattern, content[output_directive_pos:], re.DOTALL)

            if match:
                start = output_directive_pos + match.start(1)
                end = output_directive_pos + match.end(1)

                # Get indentation for current output directive
                output_directive = next(
                    (e for e in match.group(0).split("\n") if ".. code-block:: output" in e)
                )
                output_directive_indent = output_directive.split("..")[0] + " " * 4

                # Write output to output block
                output = "\n".join([output_directive_indent + line for line in output.split("\n")])
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
                df = df.sort_values(by="file_path").reset_index(drop=True)
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
