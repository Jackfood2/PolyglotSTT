"""Self-check for itn.py (no dependencies).

Run:  python test_itn.py
"""

import itn

CASES = [
    # --- digits spoken one by one (the "9950" bug) -----------------------
    ("nine nine five zero", "9950"),
    ("my order number is nine nine five zero", "my order number is 9950"),
    ("Nine nine five zero.", "9950."),
    ("the code is nine nine five zero", "the code is 9950"),
    ("nine nine five oh", "9950"),
    ("nine double nine five zero", "99950"),
    ("five nine five nine", "5959"),
    ("one nine nine five zero", "19950"),
    ("zero five nine", "059"),
    ("oh five nine", "059"),
    ("one two three four five", "12345"),
    ("extension four five six", "extension 456"),
    ("room two oh four", "room 204"),
    ("my pin is one two three four", "my pin is 1234"),
    ("double seven", "77"),
    ("triple five", "555"),
    ("oh double five", "055"),
    ("nine-nine-five-zero", "9950"),
    ("nine nine oh five", "9905"),
    ("the total was ninety nine dollars", "the total was $99"),
    # spaced digits emitted by the model itself
    ("9 9 5 0", "9950"),
    ("my code is 9 9 5 0", "my code is 9950"),
    ("2 4 6 8", "2468"),
    ("3 30", "3 30"),
    # --- clock times ----------------------------------------------------
    ("three thirty pm", "3:30 PM"),
    ("meet at three thirty", "meet at 3:30"),
    ("three thirty", "3:30"),
    ("at three oh five pm", "at 3:05 PM"),
    ("three oh five", "3:05"),
    ("three o'clock", "3:00"),
    ("seven forty", "7:40"),
    ("ten fifteen", "10:15"),
    # --- years ----------------------------------------------------------
    ("in twenty twenty four", "in 2024"),
    ("since nineteen ninety", "since 1990"),
    ("in nineteen eighty four", "in 1984"),
    ("the year twenty twenty five", "the year 2025"),
    # --- cardinals, money, percent, decimals, ordinals -------------------
    ("twenty five hundred", "2500"),
    ("a hundred and twenty three", "123"),
    ("one hundred and five percent", "105%"),
    ("two thousand and twenty four dollars", "$2024"),
    ("minus five", "-5"),
    ("three point one four", "3.14"),
    ("point five", "0.5"),
    ("third", "3rd"),
    ("twenty-first", "21st"),
    # --- ambiguous words are left alone ---------------------------------
    ("no one", "no one"),
    ("one of them", "one of them"),
    ("the one", "the one"),
    ("wait a second", "wait a second"),
    ("first of all", "first of all"),
    ("one third", "one third"),
    ("a couple of them", "a couple of them"),
    ("chapter one two", "chapter 1 2"),
]

FAILED = 0
for text, expected in CASES:
    got = itn.normalize(text)
    if got != expected:
        FAILED += 1
        print("FAIL  %-45r -> %r  (expected %r)" % (text, got, expected))

if FAILED:
    print("\n%d of %d cases failed" % (FAILED, len(CASES)))
    raise SystemExit(1)

print("all %d ITN cases pass" % len(CASES))
