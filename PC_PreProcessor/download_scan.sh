# Author: Hugo Nilsson
# Date: 2026-01-30
# Simple bash script to download .ply scan from google drive using gdown.
# Important: Set google drive folder permissions to "Anyone with the link".
#
# Usage: ./download_scan.sh <google_drive_folder_url> [output_filename] 
#
#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <google_drive_folder_url> [output_filename]"
    exit 1
fi

SRC="$1"
OUTPUT="${2:-}"

DEST_DIR="./scans"

# Downloads gdown if missing
if ! command -v gdown >/dev/null 2>&1; then
  echo "Installing gdown..."
  python3 -m pip install --user -U gdown
  export PATH="$HOME/.local/bin:$PATH"
fi
# gdown download
if [[ -n "$OUTPUT" ]]; then
    gdown --fuzzy "$SRC" -O "$DEST_DIR/$OUTPUT"
else
    gdown --fuzzy "$SRC"
fi

echo "Download completed."