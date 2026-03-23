"""
Student Name : Eaint Taryar Linlat 
Key Summary – Task: The "Date Drift" Amortization Audit
In this task, I built a daily simple interest calculator that correctly handles irregular payment dates, including leap year edge cases — something a standard LLM often gets wrong.
What I did:

Implemented calculate_irregular_interest() — computes interest using the Actual/365 method: interest = balance × (0.12 / 365) × actual_days_elapsed where days are counted precisely between each payment date using Python's datetime subtraction
Handled the leap day correctly — Feb 29, 2024 exists because 2024 is a leap year, so the period from Jan 31 to Feb 29 is exactly 29 days, not 28 or 30
Tracked the running balance row by row — each period starts from the previous ending balance, accrues interest over the exact day count, then subtracts the payment
Printed a full amortization table showing date, days elapsed, starting balance, interest accrued, balance before payment, payment, and ending balance

Key lessons:

Actual/365 vs 30/360 — using real calendar days instead of assuming 30-day months produces different (and more accurate) interest figures; the difference compounds across multiple periods
LLMs hallucinate smooth tables — a naive prompt will assume equal 30-day months, miss the leap day entirely, and produce a wrong ending balance that looks plausible but is incorrect
datetime subtraction is the safest approach — Python handles leap years, month-length differences, and year boundaries automatically, removing the need for manual day-count logic that is easy to get wrong"""from datetime import datetime

def calculate_irregular_interest(principal, annual_rate, start_date, payments):
    """Compute running balance for each payment using Actual/365 daily simple accrual."""
    balance = float(principal)
    daily_rate = annual_rate / 365.0
    history = []
    prev_date = datetime.strptime(start_date, "%Y-%m-%d")

    for payment_date_str, payment_amount in payments:
        this_date = datetime.strptime(payment_date_str, "%Y-%m-%d")
        if this_date <= prev_date:
            raise ValueError("Payment date must be after prior date")

        days = (this_date - prev_date).days
        period_interest = balance * daily_rate * days
        balance_before = balance + period_interest
        balance_after = balance_before - payment_amount

        history.append({
            'payment_date': payment_date_str,
            'days_elapsed': days,
            'starting_balance': round(balance, 2),
            'interest': round(period_interest, 2),
            'balance_before_payment': round(balance_before, 2),
            'payment': round(payment_amount, 2),
            'ending_balance': round(balance_after, 2)
        })

        balance = balance_after
        prev_date = this_date

    return history


def print_amortization_summary(history):
    print(f"{'Date':10} {'Days':>4} {'Start':>12} {'Interest':>10} {'Before':>12} {'Payment':>10} {'End':>12}")
    for row in history:
        print(
            f"{row['payment_date']:10} "
            f"{row['days_elapsed']:4d} "
            f"{row['starting_balance']:12.2f} "
            f"{row['interest']:10.2f} "
            f"{row['balance_before_payment']:12.2f} "
            f"{row['payment']:10.2f} "
            f"{row['ending_balance']:12.2f}"
        )
    print(f"Final balance: {history[-1]['ending_balance']:.2f}")


if __name__ == '__main__':
    principal = 100000.0
    annual_rate = 0.12
    start_date = '2024-01-01'
    payments = [
        ('2024-01-31', 5000.0),
        ('2024-02-29', 5000.0),
        ('2024-03-31', 5000.0)
    ]

    hist = calculate_irregular_interest(principal, annual_rate, start_date, payments)
    print_amortization_summary(hist)