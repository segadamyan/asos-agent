from contextlib import contextmanager
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent


def load_prompt(name: str) -> str:
    """
    Load a prompt from the prompts directory.

    Args:
        name (str): The name of the prompt file (without extension).

    Returns:
        str: The content of the prompt file.
    """
    path = BASE_PATH / "prompts" / f"{name}.md"
    try:
        with open(str(path), "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt '{path}' not found in prompts directory.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the prompt: {e}")


@contextmanager
def collect_exception(errors: list):
    try:
        yield
    except Exception as e:
        errors.append(e)
        raise
