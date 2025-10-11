#!/bin/bash
# Assume you're in the ~/somethingsomethingV2 directory

# 1. Define the base URL (the common prefix of the URL you provided)
BASE="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2"

# 2. List of segment names (there are about 20 segments according to the official source; fill them all)
#    Below only the first two are listed as examples; you should add all "00 .. 19"
segments=(
    "20bn-something-something-v2-00"
    "20bn-something-something-v2-01"
)

# 3. Download each segment
for seg in "${segments[@]}"; do
    echo "Downloading ${seg} ..."
    # Use -L and -O so curl follows redirects and saves the file using its original name
    curl -L -O "${BASE}/${seg}"
done

# 4. Check for corrupted or missing segments (compare MD5 or file size)
#    (If the official source provides md5 or checksum, compare against it)

# 5. Combine and extract
#    This command concatenates all segments and pipes to tar for extraction
echo "Combining and extracting ..."
cat 20bn-something-something-v2-* | tar zx

# 6. Download labels / annotations
echo "Downloading labels / annotations ..."
curl -L -O https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip

# (Place the downloaded JSON into your intended annotation directory)
