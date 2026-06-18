import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from code import validate_loan_amount, score_loan


def test_negative_loan_amount_is_rejected():
    with pytest.raises(ValueError):
        validate_loan_amount(-5000)


def test_zero_loan_amount_is_rejected():
    with pytest.raises(ValueError):
        validate_loan_amount(0)


def test_negative_loan_amount_not_scored():
    with pytest.raises(ValueError):
        score_loan(-5000)


def test_valid_loan_amount_passes_validation():
    assert validate_loan_amount(5000) == 5000.0


def test_valid_loan_amount_is_scored():
    result = score_loan(5000)
    assert result["loan_amount"] == 5000.0
    assert result["approved"] is True
