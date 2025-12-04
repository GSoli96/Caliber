#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to integrate monitoring helpers into relational_profiling_tab.py
"""

def integrate_helpers():
    # Read the main file
    with open(r'c:\Users\giand\Desktop\Demo_EDBT\GUI\relational_profiling_tab.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Read the helper functions
    with open(r'c:\Users\giand\Desktop\Demo_EDBT\GUI\_monitoring_helpers.py', 'r', encoding='utf-8') as f:
        helpers_content = f.read()
    
    # Find the insertion point (after line 418 - after zscore_outlier_rate)
    insert_index = None
    for i, line in enumerate(lines):
        if 'return float((zvals.abs() > z).mean())' in line:
            insert_index = i + 1
            break
    
    if insert_index is None:
        print("ERROR: Could not find insertion point")
        return False
    
    # Insert the helpers
    new_lines = lines[:insert_index] + ['\n', helpers_content, '\n'] + lines[insert_index:]
    
    # Write back
    with open(r'c:\Users\giand\Desktop\Demo_EDBT\GUI\relational_profiling_tab.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"Successfully integrated helpers at line {insert_index}")
    return True

if __name__ == '__main__':
    integrate_helpers()
