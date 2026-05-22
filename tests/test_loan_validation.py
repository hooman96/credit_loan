import pytest
from code import validate_loan_amount


class TestValidateLoanAmount:
    """Regression tests for loan amount validation."""
    
    def test_validate_loan_amount_negative_integer(self):
        """Test that negative loan amounts raise ValueError."""
        with pytest.raises(ValueError, match="Loan amount cannot be negative"):
            validate_loan_amount(-100)
    
    def test_validate_loan_amount_negative_float(self):
        """Test that negative float loan amounts raise ValueError."""
        with pytest.raises(ValueError, match="Loan amount cannot be negative"):
            validate_loan_amount(-0.01)
    
    def test_validate_loan_amount_zero(self):
        """Test that zero loan amount is accepted."""
        result = validate_loan_amount(0)
        assert result == 0
    
    def test_validate_loan_amount_positive_integer(self):
        """Test that positive integer loan amounts are accepted."""
        result = validate_loan_amount(1000)
        assert result == 1000
    
    def test_validate_loan_amount_positive_float(self):
        """Test that positive float loan amounts are accepted."""
        result = validate_loan_amount(0.01)
        assert result == 0.01
    
    def test_validate_loan_amount_large_positive(self):
        """Test that large positive loan amounts are accepted."""
        result = validate_loan_amount(1_000_000.50)
        assert result == 1_000_000.50
