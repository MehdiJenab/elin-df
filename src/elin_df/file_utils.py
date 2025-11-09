"""
file_utils.py - Utility functions for file operations in elin_df package

This module provides reusable file handling functions to avoid code duplication.
"""

import os


def remove_trailing_comma(filepath, trailing=', '):
    """
    Remove trailing characters from a file.
    
    Parameters
    ----------
    filepath : str
        Path to the file to modify
    trailing : str, optional
        The trailing string to remove (default: ', ')
    
    Returns
    -------
    bool
        True if trailing string was found and removed, False otherwise
    
    Examples
    --------
    >>> remove_trailing_comma('output.json')
    True
    >>> remove_trailing_comma('output.json', trailing=',\n')
    False
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if content.endswith(trailing):
        content = content[:-len(trailing)]
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False


def append_to_file(filepath, text):
    """
    Append text to a file.
    
    Parameters
    ----------
    filepath : str
        Path to the file
    text : str
        Text to append
    
    Examples
    --------
    >>> append_to_file('output.json', '\\n]\\n}')
    """
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(text)


def write_json_section_end(filepath, indent_level=2):
    """
    Write the standard JSON section ending (], },).
    
    Parameters
    ----------
    filepath : str
        Path to the JSON file
    indent_level : int, optional
        Number of tab indentation levels (default: 2)
    
    Examples
    --------
    >>> write_json_section_end('output.json', indent_level=2)
    """
    tab = "\t"
    stg = "\n"
    stg += tab * indent_level + "]" + "\n"
    stg += tab * (indent_level - 1) + "}," + "\n\n"
    append_to_file(filepath, stg)


def get_script_directory(file_path):
    """
    Get the directory containing the given file.
    
    Parameters
    ----------
    file_path : str
        Usually __file__ from the calling script
    
    Returns
    -------
    str
        Absolute path to the directory
    
    Examples
    --------
    >>> script_dir = get_script_directory(__file__)
    >>> config_path = os.path.join(script_dir, 'config.json')
    """
    return os.path.dirname(os.path.abspath(file_path))


def safe_file_operation(filepath, operation='r', **kwargs):
    """
    Context manager for safe file operations with proper error handling.
    
    Parameters
    ----------
    filepath : str
        Path to the file
    operation : str, optional
        File mode ('r', 'w', 'a', etc.)
    **kwargs
        Additional arguments passed to open()
    
    Yields
    ------
    file object
        Opened file handle
    
    Examples
    --------
    >>> with safe_file_operation('data.json', 'r') as f:
    ...     data = f.read()
    """
    try:
        f = open(filepath, operation, **kwargs)
        yield f
    except Exception as e:
        print(f"Error with file {filepath}: {e}")
        raise
    finally:
        if 'f' in locals():
            f.close()


def ensure_directory_exists(filepath):
    """
    Ensure the directory for a file path exists, create if necessary.
    
    Parameters
    ----------
    filepath : str
        Path to a file
    
    Examples
    --------
    >>> ensure_directory_exists('/path/to/output/file.json')
    """
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
