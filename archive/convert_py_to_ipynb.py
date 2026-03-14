#!/usr/bin/env python3
"""
Convert train_v2_improved.py to train_v2_improved.ipynb
"""

import json
import re

# Read the Python file
with open('train_v2_improved.py', 'r') as f:
    py_content = f.read()

# Split into sections based on comments
sections = []
current_section = {'type': 'markdown', 'content': ''}
code_buffer = []

lines = py_content.split('\n')
i = 0

# Skip shebang and initial docstring
while i < len(lines):
    if lines[i].startswith('#!/') or lines[i].startswith('"""'):
        i += 1
        if '"""' in lines[i-1]:
            # Multi-line docstring
            while i < len(lines) and '"""' not in lines[i]:
                i += 1
            i += 1  # Skip closing """
    else:
        break

# Process the rest
while i < len(lines):
    line = lines[i]

    # Check if this is a section header (# ====... or similar)
    if line.startswith('# ' + '='*70) or line.startswith('# ='*35):
        # Save previous code section if exists
        if code_buffer:
            sections.append({
                'type': 'code',
                'content': '\n'.join(code_buffer)
            })
            code_buffer = []

        # Get the section title (next line)
        i += 1
        if i < len(lines) and lines[i].startswith('#'):
            title = lines[i].lstrip('#').strip()
            sections.append({
                'type': 'markdown',
                'content': f'## {title}'
            })
        i += 1

        # Skip the closing ===
        if i < len(lines) and (lines[i].startswith('# ' + '='*70) or lines[i].startswith('# ='*35)):
            i += 1
        continue

    # Regular code line
    code_buffer.append(line)
    i += 1

# Add final code section
if code_buffer:
    sections.append({
        'type': 'code',
        'content': '\n'.join(code_buffer)
    })

# Create notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add title cell
notebook['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Dementia Assessment Training V2 - IMPROVED\n",
        "\n",
        "**🚀 This version includes advanced techniques to fix the class prediction problem!**\n",
        "\n",
        "## What's New in V2\n",
        "\n",
        "✅ **Class Weighting**: 2x penalty for misclassifying dementia\n",
        "✅ **Oversampling**: Balanced 50/50 training set\n",
        "✅ **Better Metrics**: F1-macro + per-class recall\n",
        "✅ **Data Augmentation**: Noise, time stretch, pitch shift\n",
        "✅ **Longer Audio**: 15 seconds instead of 10\n",
        "✅ **More Epochs**: 15 instead of 10\n",
        "\n",
        "## Previous Problem\n",
        "\n",
        "The original model only predicted \"nodementia\" (77% accuracy but 0% recall for dementia).\n",
        "This version forces the model to learn BOTH classes.\n",
        "\n",
        "**See [TRAINING_ANALYSIS.md](TRAINING_ANALYSIS.md) for detailed problem analysis**"
    ]
})

# Process sections
for section in sections:
    if section['content'].strip():
        if section['type'] == 'markdown':
            notebook['cells'].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [section['content']]
            })
        else:
            # Split code into logical chunks
            code = section['content'].strip()

            # Remove log file setup and logging function
            if 'log_file =' in code or 'def log(' in code:
                # Replace log() calls with print()
                code = code.replace('log(', 'print(')
                # Remove log_file definition and log function
                code_lines = code.split('\n')
                filtered_lines = []
                skip_until_next_section = False
                for line in code_lines:
                    if 'log_file =' in line or 'def log(' in line:
                        skip_until_next_section = True
                    elif skip_until_next_section and (line.strip() == '' or not line.startswith(' ')):
                        skip_until_next_section = False
                    elif not skip_until_next_section and 'with open(log_file' not in line:
                        filtered_lines.append(line)
                code = '\n'.join(filtered_lines)

            # Remove any remaining log file writes
            code = re.sub(r"\s*with open\(log_file.*?\n.*?f\.write.*?\n", "", code, flags=re.DOTALL)

            if code.strip():
                # Convert code lines to array format
                code_lines = code.split('\n')
                source_array = [line + '\n' if i < len(code_lines) - 1 else line
                               for i, line in enumerate(code_lines)]

                notebook['cells'].append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source_array
                })

# Write notebook
with open('train_v2_improved.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("="*70)
print("✅ NOTEBOOK CREATED!")
print("="*70)
print("\nFile: train_v2_improved.ipynb")
print("\nConversion complete:")
print(f"  - {len(notebook['cells'])} cells created")
print(f"  - Markdown cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')}")
print(f"  - Code cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')}")
print("\nChanges made:")
print("  - log() calls replaced with print()")
print("  - Log file setup removed (notebook has built-in output)")
print("  - Sections organized into cells")
print("\nReady to run in Jupyter!")
print("="*70)
