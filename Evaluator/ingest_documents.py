import sys
from pathlib import Path

from rag import ingest_documents_to_vector_store


SUPPORTED_EXTENSIONS = {
    ".md",
    ".txt",
}


def get_files_from_directory(directory_path: str) -> list[str]:
    """
    מחזיר את כל הקבצים הנתמכים מתוך תיקייה.
    """

    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {directory_path}"
        )

    if not directory.is_dir():
        raise ValueError(
            f"Path is not a directory: {directory_path}"
        )

    files = []

    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(file))

    return sorted(files)


def main():
    """
    Entry point
    """

    if len(sys.argv) < 2:
        print(
            "Usage: python ingest_documents.py <directory_path>"
        )
        sys.exit(1)

    directory_path = sys.argv[1]

    try:
        files = get_files_from_directory(directory_path)

        if not files:
            print("No supported files found.")
            return

        print(f"Found {len(files)} files")

        for file in files:
            print(f" - {file}")

        ingest_documents_to_vector_store(files)

        print("Ingestion completed successfully")

    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()