import random

# Mappa di valori e semi
RANKS = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
SUITS = ['d','h','c','s']
RANK_MAP = {r: i+2 for i, r in enumerate(RANKS)}

# Costruisce il mazzo completo
def build_deck():
    return [(r, s) for r in RANKS for s in SUITS]

# Converte stringa tipo 'Kh' ('K','h') 
def parse_card(card_str):
    s = card_str.strip()
    if len(s) < 2:
        raise ValueError(f"Carta non valida: {card_str}")
    suit = s[-1].lower()
    rank_str = s[:-1].upper()
    # Supporto '10' proxy 'T'
    rank = 'T' if rank_str == '10' else rank_str
    if rank not in RANK_MAP or suit not in SUITS:
        raise ValueError(f"Carta non valida: {card_str}")
    return (rank, suit)

# Valutatore di mano a 7 carte
def evaluate_7cards(cards):
    counts = {}
    suits_count = {}
    for r, s in cards:
        counts[r] = counts.get(r, 0) + 1
        suits_count[s] = suits_count.get(s, 0) + 1

    # Rank in forma numerica
    all_ranks = sorted(
        [RANK_MAP[r] for r, cnt in counts.items() for _ in range(cnt)],
        reverse=True
    )
    uniq = sorted(set(all_ranks), reverse=True)

    # Aggiunge A come 1 per le scale
    seq = uniq[:]
    if 14 in seq:
        seq.append(1)
    seq = sorted(seq, reverse=True)

    # Straight
    straight_high = None
    for i in range(len(seq) - 4):
        w = seq[i:i + 5]
        if w[0] - w[4] == 4 and len(set(w)) == 5:
            straight_high = w[0]
            break

    # Flush e Straight Flush
    flush_suit = next((s for s, cnt in suits_count.items() if cnt >= 5), None)
    if flush_suit:
        flush_cards = sorted(
            [RANK_MAP[r] for r, su in cards if su == flush_suit],
            reverse=True
        )
        flush_seq = list(set(flush_cards))
        if 14 in flush_cards:
            flush_seq.append(1)
        flush_seq = sorted(flush_seq, reverse=True)
        for i in range(len(flush_seq) - 4):
            w = flush_seq[i:i + 5]
            if w[0] - w[4] == 4 and len(set(w)) == 5:
                return (8, w[0])
        return (5, *flush_cards[:5])

    # Quads, Full, Trips, Pairs
    counts_by_rank = sorted(
        ((cnt, RANK_MAP[r]) for r, cnt in counts.items()),
        reverse=True
    )
    if counts_by_rank[0][0] == 4:
        four = counts_by_rank[0][1]
        kicker = max(r for r in all_ranks if r != four)
        return (7, four, kicker)
    if counts_by_rank[0][0] == 3 and counts_by_rank[1][0] >= 2:
        return (6, counts_by_rank[0][1], counts_by_rank[1][1])
    if straight_high:
        return (4, straight_high)
    if counts_by_rank[0][0] == 3:
        three = counts_by_rank[0][1]
        kickers = [r for r in all_ranks if r != three][:2]
        return (3, three, *kickers)
    pairs = [r for cnt, r in counts_by_rank if cnt >= 2]
    if len(pairs) >= 2:
        high, low = pairs[0], pairs[1]
        kicker = next(r for r in all_ranks if r not in (high, low))
        return (2, high, low, kicker)
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = [r for r in all_ranks if r != pair][:3]
        return (1, pair, *kickers)
    return (0, *all_ranks[:5])

# Simulazione Monte Carlo equity 
def simulate_equity(hole, board, num_opponents=1, iterations=20000):
    deck = build_deck()
    deck = [c for c in deck if c not in hole + board]
    wins = ties = 0
    for _ in range(iterations):
        local = deck.copy()
        opp_hands = []
        for _ in range(num_opponents):
            hand = random.sample(local, 2)
            opp_hands.append(hand)
            for c in hand:
                local.remove(c)
        drawn = random.sample(local, 5 - len(board))
        full = board + drawn
        hero = evaluate_7cards(hole + full)
        opp = [evaluate_7cards(h + full) for h in opp_hands]
        best_opp = max(opp)
        if hero > best_opp:
            wins += 1
        elif hero == best_opp:
            ties += 1
    return wins/iterations*100, ties/iterations*100

# Main script
if __name__ == '__main__':
    hole = [parse_card(x) for x in input("Hole cards: ").split()] # (es. 5d Kh)
    board = []

    # Pre-flop
    n0 = int(input("Avversari pre-flop: "))
    w, t = simulate_equity(hole, board, num_opponents=n0)
    print(f"Pre-flop: Vittoria ~{w:.2f}% | Tie ~{t:.2f}%\n")

    # Flop
    board = [parse_card(x) for x in input("Flop (3 carte): ").split()]
    n1 = int(input("Avversari dopo il flop: "))
    w, t = simulate_equity(hole, board, num_opponents=n1)
    print(f"Post-flop: Vittoria ~{w:.2f}% | Tie ~{t:.2f}%\n")

    # Turn
    board.append(parse_card(input("Turn (1 carta): ")))
    n2 = int(input("Avversari dopo il turn: "))
    w, t = simulate_equity(hole, board, num_opponents=n2)
    print(f"Post-turn: Vittoria ~{w:.2f}% | Tie ~{t:.2f}%\n")

    # River
    board.append(parse_card(input("River (1 carta): ")))
    n3 = int(input("Avversari dopo il river: "))
    w, t = simulate_equity(hole, board, num_opponents=n3)
    print(f"Post-river: Vittoria ~{w:.2f}% | Tie ~{t:.2f}%")
