#!/bin/bash
# copy_web_python_files.sh
# Recursively copy only Python and web files (.py, .html, .css, .js) to clipboard

OUTPUT=""

# Find all relevant files recursively
while IFS= read -r file; do
    OUTPUT+="=== $file ==="$'\n'
    OUTPUT+="$(cat "$file")"$'\n\n'
done < <(find . -type f \( -name "*.py" -o -name "*.html" -o -name "*.css" -o -name "*.js" \) | sort)

# Copy to clipboard
echo "$OUTPUT" | pbcopy
echo "Copied all Python and web files recursively to clipboard!"

