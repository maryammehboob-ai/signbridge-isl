"""Turn the letters recognised by the camera into meaningful words and sentences.

Fingerspelling is read one letter at a time, so a word can contain a wrong or missing letter
(HELO instead of HELLO). This module checks the word being built against an English dictionary
(offline, no internet) and suggests corrections. Nothing is changed unless you click a suggestion
(or switch on the "fix spelling when I press Space" option).

Names (PRIYA) are usually not in the dictionary: they get a note, never a forced change.
"""


MIN_FIX_FREQUENCY = 1e-4          # a correction must be a COMMON word (about 1 in 10,000 words) to be applied automatically


class Speller:
    def __init__(self):
        self._spell = None

    def _dictionary(self):
        if self._spell is None:
            from spellchecker import SpellChecker
            self._spell = SpellChecker()
        return self._spell

    @staticmethod
    def _checkable(word):
        word = (word or "").strip()
        return len(word) >= 2 and word.isalpha()

    def is_word(self, word):
        """True for a dictionary word. Numbers and mixed text (A1B2) are never 'misspelled'."""
        word = (word or "").strip().lower()
        if not word:
            return True
        if not word.isalpha():
            return True
        if len(word) == 1:
            return word in ("a", "i")
        return word in self._dictionary()

    def suggestions(self, word, limit=5):
        """Likely intended words (upper case), closest first, then the most common. Empty if the word is fine."""
        if not self._checkable(word) or self.is_word(word):
            return []
        spell = self._dictionary()
        low = word.strip().lower()
        candidates = spell.candidates(low) or set()
        candidates.discard(low)
        one_edit = spell.edit_distance_1(low)
        ranked = sorted(candidates, key=lambda c: (0 if c in one_edit else 1, -spell.word_usage_frequency(c), c))
        return [c.upper() for c in ranked[:limit]]

    def best_fix(self, word):
        """One safe automatic correction, or None. Only a single-letter mistake is fixed, and only
        when the dictionary has a clearly most common candidate."""
        if not self._checkable(word) or self.is_word(word):
            return None
        spell = self._dictionary()
        low = word.strip().lower()
        one_edit = spell.edit_distance_1(low)
        options = sorted((c for c in (spell.candidates(low) or ()) if c in one_edit and c != low),
                         key=lambda c: -spell.word_usage_frequency(c))
        if not options or spell.word_usage_frequency(options[0]) < MIN_FIX_FREQUENCY:
            return None                                        # only common words: names (PRIYA -> ORIYA) are left alone
        if len(options) > 1 and spell.word_usage_frequency(options[0]) < 2 * spell.word_usage_frequency(options[1]):
            return None                                        # two similar candidates: let the person choose
        return options[0].upper()


def readable_sentence(text):
    """HELLO MY NAME IS -> 'Hello my name is' (first letter capital, I capital)."""
    words = (text or "").lower().split()
    words = ["I" if w == "i" else w for w in words]
    if words and words[0] != "I":
        words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)
