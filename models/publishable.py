from typing import List, Tuple

def analyze_publishability(bold_words: List[str]) -> Tuple[str, int]:
    """Analyze if a paper is publishable based on bold text analysis"""
    number_words = [word for word in bold_words if any(char.isdigit() for char in word)]
    if not number_words:
        return "non-publishable", 0

    total_count = len(number_words)
    last_number_start = int(next((char for char in number_words[-1] if char.isdigit()), 0))
    difference = abs(last_number_start - total_count)
    return ("publishable" if difference >= 2 else "non-publishable", difference)
