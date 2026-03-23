#!/bin/sh
# Comment the following in the unitree setup.py:
#       install_requires=[
#             # "cyclonedds==0.10.2",
#             # "numpy",
#             # "opencv-python",
#       ],

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
file="$SCRIPT_DIR/../src/unitree_sdk2_python/setup.py"
tmp="$(mktemp)"

awk '
BEGIN {
    in_install_requires = 0
}

{
    if ($0 ~ /install_requires=\[/) {
        in_install_requires = 1
    }

    if (in_install_requires) {
        if ($0 ~ /^[[:space:]]*"cyclonedds==0\.10\.2",[[:space:]]*$/ ||
            $0 ~ /^[[:space:]]*"numpy",[[:space:]]*$/ ||
            $0 ~ /^[[:space:]]*"opencv-python",[[:space:]]*$/) {
            sub(/^([[:space:]]*)/, "&# ")
        }
    }

    print

    if (in_install_requires && $0 ~ /^[[:space:]]*\],[[:space:]]*$/) {
        in_install_requires = 0
    }
}
' "$file" > "$tmp"

mv "$tmp" "$file"

# bruh, typo in the __init__.py is unitree a million dollar company???
mv "$SCRIPT_DIR/./__init__.py" "$SCRIPT_DIR/./../src/unitree_sdk2_python/unitree_sdk2py/__init__.py"
