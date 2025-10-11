#!/bin/bash
# Ensure the script stops immediately on errors
set -e

# --- Configuration ---
# Base URL for dataset segment files
BASE_URL="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2"
# Prefix for segment files
SEGMENT_PREFIX="20bn-something-something-v2"
# URL for label files
LABEL_URL="https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip"
LABEL_ZIP="20bn-something-something-download-package-labels.zip"

# --- Color Definitions ---
C_BLUE='\033[0;34m'
C_GREEN='\033[0;32m'
C_YELLOW='\033[0;33m'
C_NONE='\033[0m' # No color

# --- Helper Functions ---
# Print a colored header
print_header() {
    echo -e "\n${C_BLUE}===================================================${C_NONE}"
    echo -e "${C_BLUE} $1 ${C_NONE}"
    echo -e "${C_BLUE}===================================================${C_NONE}"
}

# --- Main Functions ---

# 1. Download all dataset segments
download_segments() {
    print_header "1. Starting download of dataset segments (20 total)"
    # Use a loop to generate filenames from 00 to 19
    for i in $(seq -f "%02g" 0 19); do
        SEGMENT_NAME="${SEGMENT_PREFIX}-${i}"
        DOWNLOAD_URL="${BASE_URL}/${SEGMENT_NAME}"
        echo -e "${C_YELLOW} -> Downloading ${SEGMENT_NAME}...${C_NONE}"
        # -L: follow redirects, -O: save with remote filename, -sS: silent but show errors
        curl -L -O -sS "${DOWNLOAD_URL}"
    done
    echo -e "${C_GREEN}✔ All segments downloaded.${C_NONE}"
}

# 2. Merge and extract segment files
extract_segments() {
    print_header "2. Merging and extracting segment files"
    echo -e "${C_YELLOW} -> Merging all segments and extracting via 'tar'...${C_NONE}"
    # Merge all segment files in order and pipe to tar for extraction
    cat ${SEGMENT_PREFIX}-* | tar zx
    echo -e "${C_GREEN}✔ Extraction completed.${C_NONE}"

    echo -e "${C_YELLOW} -> Deleting downloaded segment files...${C_NONE}"
    rm ${SEGMENT_PREFIX}-*
    echo -e "${C_GREEN}✔ Segment files deleted.${C_NONE}"
}

# 3. Download and extract label files
download_labels() {
    print_header "3. Downloading and extracting label/annotation files"
    echo -e "${C_YELLOW} -> Downloading label file...${C_NONE}"
    curl -L -o "${LABEL_ZIP}" -sS "${LABEL_URL}"
    echo -e "${C_GREEN}✔ Label file downloaded.${C_NONE}"

    echo -e "${C_YELLOW} -> Extracting label file...${C_NONE}"
    unzip -q "${LABEL_ZIP}" # -q: quiet mode
    echo -e "${C_GREEN}✔ Label file extracted.${C_NONE}"

    echo -e "${C_YELLOW} -> Deleting label ZIP file...${C_NONE}"
    rm "${LABEL_ZIP}"
    echo -e "${C_GREEN}✔ Label ZIP file deleted.${C_NONE}"
}

# --- Execute Script ---
main() {
    download_segments
    extract_segments
    download_labels
    print_header "All tasks completed successfully!"
}

# Run main function
main
