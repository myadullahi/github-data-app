"""
Shared rules for which repo files count as "eligible" (text + code we load).
Used by discover_repos (to count repo size) and load_data (to fetch).
"""
from pathlib import Path

MAX_FILE_BYTES = 100_000
SKIP_PATH_PARTS = (
    "node_modules", "vendor", "__pycache__", ".git", "dist", "build", ".next",
    "venv", ".venv", "env", ".eggs", "*.egg-info", "target", "out", ".cache",
)
SKIP_PATH_SUFFIXES = (".min.js", ".min.css", "-min.js", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", ".map")
TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".js", ".ts", ".jsx", ".tsx", ".vue", ".css", ".scss", ".html", ".htm",
    ".sh", ".bash", ".zsh", ".sql", ".r", ".rb", ".go", ".rs", ".java", ".kt",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".m", ".mm",
}


def should_skip_path(path: str) -> bool:
    path_lower = path.lower().replace("\\", "/")
    for part in SKIP_PATH_PARTS:
        if f"/{part}/" in path_lower or path_lower.startswith(part + "/") or path_lower == part:
            return True
    for suf in SKIP_PATH_SUFFIXES:
        if path_lower.endswith(suf):
            return True
    return False


def is_text_file(path: str, size: int) -> bool:
    if size > MAX_FILE_BYTES:
        return False
    if should_skip_path(path):
        return False
    ext = Path(path).suffix.lower()
    return ext in TEXT_EXTENSIONS or ext == ""
