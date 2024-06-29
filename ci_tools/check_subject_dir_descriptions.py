import fnmatch
import pathlib
import sys
from typing import Iterable, Iterator

import marko.block
import marko.element
import marko.inline

README_FILE = "README.md"
EXCLUDE_DIRS = ("ci_tools", r".*")


def iterate_elements(root: marko.block.BlockElement) -> Iterator[marko.element.Element]:
    """Recursively iterate through all elements in the Markdown document."""
    for child in root.children:
        yield child
        if isinstance(child, marko.block.BlockElement):
            yield from iterate_elements(child)


def extract_links(elements: Iterable[marko.element.Element]) -> Iterator[str]:
    """Extract all links from the given elements."""
    for element in elements:
        if isinstance(element, marko.inline.Link):
            yield element.dest


def filter_existing_paths(links: Iterable[str]) -> Iterator[str]:
    """Filter links that point to existing paths."""
    for link in links:
        if pathlib.Path(link).exists():
            yield link


def should_exclude_directory(path: pathlib.Path) -> bool:
    """Check if the directory should be excluded based on patterns."""
    path_repr = str(path)
    return any(fnmatch.fnmatch(path_repr, pattern) for pattern in EXCLUDE_DIRS)


def get_subject_directories() -> Iterator[pathlib.Path]:
    """Retrieve all subject directories, excluding specified ones."""
    for path in pathlib.Path(".").glob("*"):
        if path.is_dir() and not should_exclude_directory(path):
            yield path


def main():
    # Get subject directories
    subject_dirs = {str(path) for path in get_subject_directories()}

    # Parse README file
    readme_path = pathlib.Path(README_FILE)
    readme_content = readme_path.read_text()
    readme_document = marko.parse(readme_content)

    # Extract links from README file
    links = extract_links(iterate_elements(readme_document))
    linked_paths = {str(pathlib.Path(link)) for link in filter_existing_paths(links)}

    # Find directories without links in README
    dirs_without_links = subject_dirs - linked_paths
    if dirs_without_links:
        print(
            f"error: following subject directories not mentioned: {', '.join(dirs_without_links)}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
