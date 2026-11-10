import pytest

from fundoscope.sentinel import CreditBudget, batch_cap


@pytest.mark.parametrize(("used", "cap"), [(0, 8_000), (7_900, 100), (8_000, 0), (9_500, 0)])
def test_batch_cap_is_80_percent_of_the_month_minus_what_was_used(used, cap):
    assert batch_cap(used) == cap


def test_batch_cap_rejects_negative_usage():
    with pytest.raises(ValueError, match="can't be negative"):
        batch_cap(-1)


def test_requests_start_only_while_they_fit_under_the_cap():
    budget = CreditBudget(25)

    first, second = budget.reserve(), budget.reserve()

    assert (first, second) == (10, 10)
    assert budget.reserve() is None  # 20 held + 10 expected > 25


def test_settling_records_what_was_actually_spent():
    budget = CreditBudget(25)
    hold = budget.reserve()

    budget.settle(hold, 7.0)

    assert (budget.spent, budget.held) == (7.0, 0.0)
    assert budget.reserve() == 10  # 7 + 10 <= 25
    assert budget.reserve() is None  # 7 + 10 held + 10 > 25


def test_an_expensive_request_raises_the_expected_cost():
    budget = CreditBudget(100)
    budget.settle(budget.reserve(), 30.0)
    assert budget.expected == 30.0
    assert budget.reserve() == 30.0
