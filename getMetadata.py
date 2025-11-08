#!/usr/bin/env python3
import sys
from pngmeta import PngMeta

if len(sys.argv) > 1:
    file_name = sys.argv[1]
else:
    sys.exit(1)

meta = PngMeta(file_name)
for key, value in meta.items():
    print(f'{key}: {value}')
