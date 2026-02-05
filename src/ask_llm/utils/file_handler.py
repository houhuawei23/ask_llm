"""File handling utilities with progress bars."""

from pathlib import Path
from typing import Optional, Union

from loguru import logger
from tqdm import tqdm


class FileHandler:
    """Handle file I/O operations with progress tracking."""

    CHUNK_SIZE = 8192  # 8KB chunks for reading

    @classmethod
    def read(cls, path: Union[str, Path], show_progress: bool = False) -> str:
        """
        Read content from a file.

        Args:
            path: Path to the file
            show_progress: Whether to show progress bar

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        try:
            size = file_path.stat().st_size

            if show_progress and size > cls.CHUNK_SIZE:
                content = cls._read_with_progress(file_path, size)
            else:
                content = file_path.read_text(encoding="utf-8")

            logger.debug(f"Read {len(content)} characters from {path}")
            return content

        except UnicodeDecodeError as e:
            raise OSError(f"File {path} is not valid UTF-8 text: {e}") from e
        except Exception as e:
            raise OSError(f"Failed to read file {path}: {e}") from e

    @classmethod
    def _read_with_progress(cls, path: Path, total_size: int) -> str:
        """Read file with progress bar."""
        content_parts = []

        with open(path, encoding="utf-8") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=f"Reading {path.name}", ncols=80
        ) as pbar:
            while True:
                chunk = f.read(cls.CHUNK_SIZE)
                if not chunk:
                    break
                content_parts.append(chunk)
                pbar.update(len(chunk.encode("utf-8")))

        return "".join(content_parts)

    @classmethod
    def write(
        cls, path: Union[str, Path], content: str, force: bool = False, show_progress: bool = False
    ) -> None:
        """
        Write content to a file.

        Args:
            path: Path to the output file
            content: Content to write
            force: Whether to overwrite existing file
            show_progress: Whether to show progress bar

        Raises:
            FileExistsError: If file exists and force is False
            IOError: If file cannot be written
        """
        file_path = Path(path)

        # Check if file exists
        if file_path.exists() and not force:
            raise FileExistsError(f"Output file already exists: {path}. Use --force to overwrite.")

        # Create parent directories
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f"Failed to create directory for {path}: {e}") from e

        try:
            if show_progress and len(content) > cls.CHUNK_SIZE:
                cls._write_with_progress(file_path, content)
            else:
                file_path.write_text(content, encoding="utf-8")

            logger.debug(f"Wrote {len(content)} characters to {path}")

        except Exception as e:
            raise OSError(f"Failed to write file {path}: {e}") from e

    @classmethod
    def _write_with_progress(cls, path: Path, content: str) -> None:
        """Write file with progress bar."""
        total = len(content)
        written = 0

        with open(path, "w", encoding="utf-8") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Writing {path.name}", ncols=80
        ) as pbar:
            while written < total:
                chunk = content[written : written + cls.CHUNK_SIZE]
                f.write(chunk)
                written += len(chunk)
                pbar.update(len(chunk.encode("utf-8")))

    @classmethod
    def generate_output_path(
        cls,
        input_path: Union[str, Path],
        custom_path: Optional[Union[str, Path]] = None,
        suffix: str = "_output",
    ) -> str:
        """
        Generate output file path based on input path.

        Args:
            input_path: Input file path
            custom_path: Custom output path (optional)
            suffix: Suffix to add to filename

        Returns:
            Output file path
        """
        if custom_path:
            return str(custom_path)

        input_file = Path(input_path)
        ext = input_file.suffix
        stem = input_file.stem

        output_name = f"{stem}{suffix}{ext}"
        return str(input_file.parent / output_name)

    @classmethod
    def detect_type(cls, path: Union[str, Path]) -> str:
        """
        Detect file type based on extension.

        Args:
            path: File path

        Returns:
            File extension (lowercase)
        """
        return Path(path).suffix.lower()

    @classmethod
    def is_text_file(cls, path: Union[str, Path]) -> bool:
        """
        Check if file is a text file based on extension.

        Args:
            path: File path

        Returns:
            True if file appears to be text
        """
        text_extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
            ".css",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".java",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".m",
            ".mm",
            ".csv",
            ".tsv",
            ".log",
            ".ini",
            ".cfg",
            ".conf",
        }

        ext = cls.detect_type(path)
        return ext in text_extensions
