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
    print_header "1. Starting download of dataset segments (2 total)"
    # Use a loop to generate filenames from 00 to 01
    for i in $(seq -f "%02g" 0 1); do
        SEGMENT_NAME="${SEGMENT_PREFIX}-${i}"
        DOWNLOAD_URL="${BASE_URL}/${SEGMENT_NAME}"
        echo -e "${C_YELLOW} -> Downloading ${SEGMENT_NAME}...${C_NONE}"
        # -L: follow redirects, -O: save with remote filename, --progress-bar: show progress bar
        curl -L -O --progress-bar "${DOWNLOAD_URL}"
    done
    echo -e "${C_GREEN}✔ All segments downloaded.${C_NONE}"
}

# 2. Merge and extract segment files
extract_segments() {
    print_header "2. Merging and extracting segment files"

    # Check if pv (pipe viewer) is installed to show a progress bar
    if ! command -v pv &> /dev/null; then
        echo -e "${C_YELLOW} -> 'pv' not found. Merging and extracting without a progress bar.${C_NONE}"
        echo -e "${C_YELLOW}    (To see progress, you can install 'pv'. e.g., 'sudo apt-get install pv' or 'brew install pv')${C_NONE}"
        # Merge all segment files in order and pipe to tar for extraction
        cat ${SEGMENT_PREFIX}-* | tar zx
    else
        echo -e "${C_YELLOW} -> Calculating total size for progress bar...${C_NONE}"
        # Calculate the total size of all segment files for pv
        total_size=$(du -cb ${SEGMENT_PREFIX}-* | grep total | awk '{print $1}')

        echo -e "${C_YELLOW} -> Merging all segments and extracting via 'tar' with progress...${C_NONE}"
        # Merge files, pipe through pv to show progress, then pipe to tar for extraction
        cat ${SEGMENT_PREFIX}-* | pv -s $total_size | tar zx
    fi

    echo -e "${C_GREEN}✔ Extraction completed.${C_NONE}"

    echo -e "${C_YELLOW} -> Deleting downloaded segment files...${C_NONE}"
    rm ${SEGMENT_PREFIX}-*
    echo -e "${C_GREEN}✔ Segment files deleted.${C_NONE}"
}

# 3. Download and extract label files
download_labels() {
    print_header "3. Downloading and extracting label/annotation files"
    echo -e "${C_YELLOW} -> Downloading label file...${C_NONE}"
    curl -L -o "${LABEL_ZIP}" --progress-bar "${LABEL_URL}"
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
