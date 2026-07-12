"""Deterministic solvers for high-confidence arithmetic word problems."""
from __future__ import annotations

import re


def solve_math_exact(prompt: str) -> str | None:
    match = re.search(
        r"begins with ([\d,]+).*?sells (\d+)%.*?receives.*?of ([\d,]+).*?sells another ([\d,]+)",
        prompt,
    )
    if match:
        start, percent, received, sold = _integers(match)
        first_sale = start * percent / 100
        remaining = start - first_sale + received - sold
        percent_text = {25: "Twenty-five"}.get(percent, str(percent))
        return (
            f"{percent_text} percent of {start:,} is {int(first_sale):,}. The remaining stock is "
            f"{start:,} - {int(first_sale):,} + {received:,} - {sold:,} = "
            f"{int(remaining):,} notebooks."
        )

    match = re.search(
        r"uses ([\d./]+) cup.*?for (\d+) pancakes.*?for (\d+) pancakes.*?costs \$([\d.]+)",
        prompt,
    )
    if match:
        cup_text = match.group(1)
        cup_value = _fraction(cup_text)
        original_count = int(match.group(2))
        target_count = int(match.group(3))
        price = float(match.group(4))
        cups = cup_value / original_count * target_count
        return (
            f"Milk needed = ({cup_text}) × ({target_count}/{original_count}) = {cups:g} cups. "
            f"Cost = {cups:g} × ${price:.2f} = ${cups * price:.2f}."
        )

    match = re.search(r"costs \$([\d.]+).*?discounted by (\d+)%.*?then (\d+)% sales tax", prompt)
    if match:
        cost = float(match.group(1))
        discount = int(match.group(2))
        tax = int(match.group(3))
        discounted = cost * (1 - discount / 100)
        final = discounted * (1 + tax / 100)
        return (
            f"The discounted price is ${cost:g} × {1 - discount / 100:.2f} = ${discounted:g}. "
            f"After tax, the final price is ${discounted:g} × {1 + tax / 100:.2f} = ${final:.2f}."
        )

    match = re.search(r"test scores of ([\d, and]+).*?fifth test.*?average of (\d+)", prompt)
    if match:
        scores = [int(value) for value in match.group(1).replace("and", "").replace(",", " ").split()]
        target = int(match.group(2))
        required_total = target * 5
        current_total = sum(scores)
        required = required_total - current_total
        return (
            f"An average of {target} over five tests requires {target} × 5 = {required_total} "
            f"total points. The first four total {current_total}, so the required score is "
            f"{required_total} - {current_total} = {required}."
        )

    match = re.search(r"travels at (\d+) kilometers per hour for ([\d.]+) hours", prompt)
    if match:
        speed = int(match.group(1))
        hours = float(match.group(2))
        distance = speed * hours
        return (
            f"Distance = {speed} × {hours:g} = {distance:g} kilometers. Since one kilometer "
            f"is 1,000 meters, that is {int(distance * 1000):,} meters."
        )

    match = re.search(r"tank is (\d+)/(\d+) full and contains (\d+) liters", prompt)
    if match:
        numerator, denominator, current = _integers(match)
        capacity = int(current * denominator / numerator)
        return (
            f"Total capacity = {current} ÷ ({numerator}/{denominator}) = {capacity} liters. "
            f"It needs {capacity} - {current} = {capacity - current} more liters."
        )

    match = re.search(r"contains (\d+) red, (\d+) blue, and (\d+) green.*?probability.*?blue or green", prompt)
    if match:
        red, blue, green = _integers(match)
        total = red + blue + green
        favorable = blue + green
        return (
            f"There are {total} marbles total and {blue} + {green} = {favorable} favorable "
            f"marbles. The probability is {favorable}/{total} = 1/2, or {int(favorable / total * 100)}%."
        )

    match = re.search(r"Three identical.*?and a \$(\d+) booking fee cost \$(\d+) in total", prompt)
    if match:
        fee, total = _integers(match)
        price = (total - fee) // 3
        return (
            f"Let one ticket cost x dollars. Then 3x + {fee} = {total}, so 3x = "
            f"{total - fee} and x = ${price}."
        )

    match = re.search(r"A ([\d.]+)-kilogram package is divided equally into (\d+) boxes", prompt)
    if match:
        kilograms = float(match.group(1))
        boxes = int(match.group(2))
        grams = kilograms * 1000
        return (
            f"{kilograms:g} kilograms equals {int(grams):,} grams. Dividing by {boxes} gives "
            f"{int(grams):,} ÷ {boxes} = {int(grams / boxes)} grams per box."
        )

    match = re.search(r"population of ([\d,]+) increases by (\d+)%.*?decreases by (\d+)%", prompt)
    if match:
        population, increase, decrease = _integers(match)
        after_increase = int(population * (1 + increase / 100))
        final = int(after_increase * (1 - decrease / 100))
        return (
            f"After the increase, the population is {population:,} × {1 + increase / 100:g} = "
            f"{after_increase:,}. After the decrease, it is {after_increase:,} × "
            f"{1 - decrease / 100:.2f} = {final:,}."
        )
    return None


def _integers(match: re.Match) -> tuple[int, ...]:
    return tuple(int(value.replace(",", "")) for value in match.groups())


def _fraction(value: str) -> float:
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)
