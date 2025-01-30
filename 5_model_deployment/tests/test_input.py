import os
import pandas as pd
import pytest

# setting path to input.csv
INPUT_FILE = "../data/input.csv"

def test_input_file_exists():
    """check input file exists"""
    assert os.path.exists(INPUT_FILE), f"Input file {INPUT_FILE} not found!"

def test_input_file_not_empty():
    """check input file not empty"""
    df = pd.read_csv(INPUT_FILE)
    assert not df.empty, "File input.csv is empty!"

def test_input_file_columns():
    """check input file has required column"""
    df = pd.read_csv(INPUT_FILE)
    required_columns = {"comment"}
    missing_columns = required_columns - set(df.columns)
    assert not missing_columns, f"Column is missed: {missing_columns}"

def test_no_empty_values():
    """Check column has no empty values"""
    df = pd.read_csv(INPUT_FILE)
    assert df["comment"].notna().all(), "Column contains missing values!"
