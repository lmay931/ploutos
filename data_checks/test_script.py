from core.verbalizer import extract_direction
cases = [
        ('The stock is expected to decline sharply.', 'DOWN'),
        ('The stock is unlikely to decline and should rise.', 'UP'),   # negation trap
        ('We expect a downward move.', 'DOWN'),
        ('A rebound higher is the base case.', 'UP'),
    ]
for text, want in cases:
    got = extract_direction(text)
    print(('OK  ' if got==want else '*** MISMATCH *** '), f'want={want} got={got} | {text}')