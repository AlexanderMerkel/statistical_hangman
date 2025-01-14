from typing import Dict, List, Set, Tuple, Union, Any
import numpy as np
import re
import string
from itertools import product
from collections import defaultdict, Counter
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna
from optuna.pruners import BasePruner

def normalize_distribution(dict_input: Dict[str, float]) -> Dict[str, float]:
    """Normalize values in a dictionary to sum to 1.0.

    Args:
        dict_input: Dictionary containing string keys and float values.

    Returns:
        Dictionary with normalized values that sum to 1.0. Returns original dict if sum is 0.
    """
    total = sum(dict_input.values())
    if total > 0:
        return {k: v/total for k, v in dict_input.items()}
    return dict_input

def round_dict(dict_input: Dict[str, float], digits: int = 3) -> Dict[str, float]:
    """Round all float values in a dictionary to specified decimal places.

    Args:
        dict_input: Dictionary containing string keys and float values.
        digits: Number of decimal places to round to.

    Returns:
        Dictionary with rounded values.
    """
    return {k: round(v, digits) for k, v in dict_input.items()}

def search_frequeny(word: str, word_list: List[str]) -> int:
    """Count occurrences of a regex pattern in a list of words.

    Args:
        word: Regular expression pattern to search for.
        word_list: List of words to search through.

    Returns:
        Number of matches found.
    """
    return sum([1 for x in word_list if re.search(word, x)])

class FeatureExtractor:
    """Extract and manage statistical features from a word list for hangman game solving.
    
    Attributes:
        max_word_length: Maximum length of words to consider.
        word_list: Array of words to analyze.
        max_ngram: Maximum length of n-grams to compute.
        max_affix_length: Maximum length of prefixes/suffixes to analyze.
        additional_length: Additional length requirement for affix analysis.
        ngram_freqs: Dictionary of n-gram frequencies.
        affix_freqs: Dictionary of prefix/suffix frequencies.
        positional_quintile_freqs: Letter frequencies by word position quintiles.
        vowel_cons_stats: Statistics about vowel/consonant patterns.
    """

    def __init__(self, 
                 word_list: np.ndarray,
                 max_word_length: int = 29,
                 max_ngram: int = 5,
                 max_affix_length: int = 6,
                 additional_length: int = 2) -> None:
        """Initialize the feature extractor with word list and parameters.

        Args:
            word_list: Array of words to analyze.
            max_word_length: Maximum length of words to consider.
            max_ngram: Maximum length of n-grams to compute.
            max_affix_length: Maximum length of prefixes/suffixes to analyze.
            additional_length: Additional length requirement for affix analysis.
        """
        self.max_word_length = max_word_length
        self.word_list = word_list
        self.max_ngram = max_ngram
        self.max_affix_length = max_affix_length
        self.additional_length = additional_length
        
        # Calculate features
        self.ngram_freqs = {}
        for n in range(1, max_ngram + 1):
            self.ngram_freqs[n] = self._calculate_ngram_freqs(n)
        
        self.affix_freqs = self._make_affix_freqs(max_affix_length, additional_length)
        self.positional_quintile_freqs = self._analyze_positional_frequencies()
        self.vowel_cons_stats = self._calculate_vowel_consonant_stats()

    def _calculate_ngram_freqs(self, n: int) -> Dict[str, int]:
        """Calculate n-gram probabilities using NLTK.

        Args:
            n: Length of n-grams to calculate.

        Returns:
            Dictionary of n-gram frequencies.
        """
        ngram_counts = dict()
        for word in self.word_list:
            if len(word) >= n:
                for ngram in ngrams(word, n):
                    joined_ngram = ''.join(ngram)
                    if joined_ngram not in ngram_counts:
                        ngram_counts[joined_ngram] = 1
                    else:
                        ngram_counts[joined_ngram] += 1
        return {ng: count for ng, count in ngram_counts.items() if count >= 1}
    
    def _make_affix_freqs(self, max_affix_length: int = 5, additional_length: int = 4) -> Dict[str, Dict[int, Counter]]:
        """Looks at prefixes and suffixes of words to find common affixes.

        Args:
            max_affix_length: Maximum length of prefixes/suffixes to analyze.
            additional_length: Additional length requirement for affix analysis.

        Returns:
            Dictionary of prefix/suffix frequencies.
        """
        affix_freqs = defaultdict(dict)
        for affix_length in range(2, max_affix_length+1):
            min_length = affix_length + additional_length
            prefix = [word[:affix_length] for word in self.word_list if len(word) >= min_length]
            suffix = [word[-affix_length:] for word in self.word_list if len(word) >= min_length]
            prefix_freqs = Counter(prefix)
            suffix_freqs = Counter(suffix)
            affix_freqs['prefix'][affix_length] = prefix_freqs
            affix_freqs['suffix'][affix_length] = suffix_freqs
        return affix_freqs

    def _analyze_positional_frequencies(self) -> List[Dict[str, int]]:
        """Analyze letter frequencies by word position quintiles.

        Returns:
            List of dictionaries with letter frequencies for each quintile.
        """
        quintile_freqs = [defaultdict(int) for _ in range(5)]
        total_words = len(self.word_list)
        
        for word in self.word_list:
            word_len = len(word)
            segment_size = word_len / 5
            
            for i, char in enumerate(word):
                quintile = min(4, int(i / segment_size))
                quintile_freqs[quintile][char] += 1
        
        return quintile_freqs

    def _calculate_vowel_consonant_stats(self) -> Dict[str, Union[float, Dict[int, float]]]:
        """Calculate statistics about vowel/consonant patterns.

        Returns:
            Dictionary with mean ratio, standard deviation, and position probabilities.
        """
        stats = {'ratios': [], 'positions': defaultdict(list)}
        vowels = set('aeiou')
        
        for word in self.word_list:
            vowel_count = sum(1 for c in word if c in vowels)
            ratio = vowel_count / len(word) if len(word) > 0 else 0
            stats['ratios'].append(ratio)
            
            for i, char in enumerate(word):
                stats['positions'][i].append(1 if char in vowels else 0)
        
        return {
            'mean_ratio': np.mean(stats['ratios']),
            'std_ratio': np.std(stats['ratios']),
            'position_probs': {
                pos: np.mean(vals) for pos, vals in stats['positions'].items()
            }
        }

    def get_matching_words(self, current_state: str, guessed_letters: Set[str]) -> np.ndarray:
        """
        Filter words that match current game state.

        Args:
            current_state: String with '_' for unknown letters.
            guessed_letters: Set of letters already guessed.

        Returns:
            np.array of matching words.
        """
        wrong_guesses = guessed_letters - set(current_state)
        matches = []
        len_matches = []

        pattern = current_state.replace('_', '.')
        compliled_pattern = re.compile(pattern)
        pattern_len = len(pattern)

        for word in self.word_list:
            if len(word) == pattern_len and compliled_pattern.match(word):
                if not wrong_guesses.intersection(set(word)):
                    matches.append(word)
        if not matches == []:
            return np.array(matches)
        else:
            return self.word_list

    def split_decision(self, word_part: str) -> bool:
        """Decide whether to split a word part based on its occurrence rate.

        Args:
            word_part: Part of the word to analyze.

        Returns:
            True if the word part should be split, False otherwise.
        """
        length = len(word_part)
        occurence_rate = sum([1 for x in self.word_list if word_part in x]) / len(self.word_list)
        if occurence_rate > (1/26) ** (len(word_part)-3):
            return True
        return False

    def decompose(self, word: str) -> str:
        """Decompose a word into prefix, infix, and suffix.

        Args:
            word: Word to decompose.

        Returns:
            Remaining part.
        """
        first_unknown = min(word.find('_'), 6)
        last_unknown = max(word.rfind('_'), len(word)-5)
        length = len(word)
        if first_unknown >= 4:
            for i in range(first_unknown, 3, -1):
                if self.split_decision(word[:i]):
                    return word[i:]
        
        if last_unknown - first_unknown >= 4:
            for i in range(length-1, last_unknown, -1):
                if self.split_decision(word[i+1:]):
                    return word[:i+1]
        return word

    def decompose2(self, word: str) -> Tuple[str, str, str]:
        """Decompose a word into prefix, infix, and suffix.

        Args:
            word: Word to decompose.

        Returns:
            Tuple containing prefix, infix, and suffix.
        """
        len_word = len(word)
        best_score = 0
        best_prefix = ''
        best_infix = ''
        best_suffix = ''
        for prefix_len in range(min(len_word, self.max_affix_length), 2, -1):
            for suffix_len in range(min(len_word - prefix_len, self.max_affix_length), 2, -1):
                prefix = word[:prefix_len]
                infix = word[prefix_len:len_word-suffix_len].replace('_', '.')
                suffix = word[-suffix_len:]
                score = self.affix_freqs['prefix'][prefix_len].get(prefix, 0) * \
                    search_frequeny(infix, self.word_list) * \
                    self.affix_freqs['suffix'][suffix_len].get(suffix, 0)
                print(prefix, infix, suffix, score)
                if score > best_score:
                    best_score = score
                    best_prefix = prefix
                    best_infix = infix
                    best_suffix = suffix
                print(prefix_len, suffix_len,
                    prefix, infix, suffix, score)
        return best_prefix, best_infix, best_suffix

class HangmanSolver:
    """Implements the strategy for solving hangman games using statistical features.
    
    Attributes:
        fe: FeatureExtractor instance providing statistical analysis.
        w_base: Weight for base letter frequency.
        w_pos: Weight for positional letter frequency.
        w_ng: Weights for n-gram frequencies.
        w_pf: Weights for prefix frequencies.
        w_sf: Weights for suffix frequencies.
        w_vowel: Weight for vowel-consonant patterns.
        atomize: Whether to use atomizing strategy for high confidence predictions.
        debug: Whether to print debug information.
    """

    def __init__(self, 
                 feature_extractor: FeatureExtractor,
                 w_base: float = 0.2,
                 w_pos: float = 0.2,
                 w_1ng: float = 0.2,
                 w_2ng: float = 0.2,
                 w_3ng: float = 0.2,
                 w_4ng: float = 0.2,
                 w_5ng: float = 0.2,
                 w_pf2: float = 0.2,
                 w_pf3: float = 0.2,
                 w_pf4: float = 0.2,
                 w_pf5: float = 0.2,
                 w_sf2: float = 0.2,
                 w_sf3: float = 0.2,
                 w_sf4: float = 0.2,
                 w_sf5: float = 0.2,
                 w_vowel: float = 0.4) -> None:
        """Initialize the solver with feature extractor and weights.

        Args:
            feature_extractor: FeatureExtractor instance.
            w_base: Weight for base letter frequency.
            w_pos: Weight for positional letter frequency.
            w_[1-5]ng: Weights for n-gram frequencies.
            w_[ps]f[2-5]: Weights for prefix/suffix frequencies.
            w_vowel: Weight for vowel-consonant patterns.
        """
        self.fe = feature_extractor
        
        # Feature component weights
        self.w_base = w_base
        self.w_pos = w_pos
        
        # N-gram weights, affix weights
        self.w_ng = [w_1ng, w_2ng, w_3ng, w_4ng, w_5ng]
        self.w_pf = [w_pf2, w_pf3, w_pf4, w_pf5]
        self.w_sf = [w_sf2, w_sf3, w_sf4, w_sf5]

        # Vowel-consonant ratio weight
        self.w_vowel = w_vowel  # Add vowel ratio weight
        self.vowels = set('aeiou')

        self.atomize = True
        
        self.debug = True

    def _get_letter_probs(self, matching_words: np.ndarray, current_state: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """Calculate probabilities for each letter based on current game state.

        Args:
            matching_words: Array of words that match the current game state.
            current_state: String with '_' for unknown letters.
            guessed_letters: Set of letters already guessed.

        Returns:
            Dictionary with letter probabilities.
        """
        unknown_positions = [i for i, c in enumerate(current_state) if c == '_']
        state_len = len(current_state)
        number_of_unknowns = len(unknown_positions)
        number_of_knowns = state_len - number_of_unknowns

        letter_counts = Counter()
        positional_counts = defaultdict(Counter)

        # Calculate base frequencies from matching words:
        # (Unigram frequencies, Unigram positional frequencies)
        # letter_counts: [letter] = frequency
        # positional_counts: [position][letter] = frequency
        for w in matching_words:
            if len(w) == state_len:
                for i in unknown_positions:
                    letter_counts[w[i]] += 1
                    positional_counts[i][w[i]] += 1

        # Base frequencies from matching words
        base_freqs = {l: letter_counts[l] for l in string.ascii_lowercase}
        if number_of_knowns == 0:
            return normalize_distribution(base_freqs)
        
        # Initialize frequencies for all letters
        letter_frequencies = {
            'base': defaultdict(float), # Format: [letter] : frequency
            'position': defaultdict(float), # Format: [letter] : frequency 
            'ngram': defaultdict(dict), # Format: [length][letter] : frequency 
            'prefix': defaultdict(dict), # Format: [length][letter] : frequency
            'suffix': defaultdict(dict), # Format: [length][letter] : frequency
            'vowel_ratio': defaultdict(float) # Format: [letter] : frequency 
        }

        # Base frequencies
        letter_frequencies['base'] = base_freqs

        for n, w in enumerate(self.w_ng, start=1):
            letter_frequencies['ngram'][n] = defaultdict(float)

        # Affix frequencies
        letter_frequencies['prefix'] = defaultdict(dict)
        letter_frequencies['suffix'] = defaultdict(dict)
        for n in range(2, self.fe.max_affix_length + 1):
            letter_frequencies['prefix'][n] = defaultdict(float)
            letter_frequencies['suffix'][n] = defaultdict(float)

        # Fill in letter frequencies based on current state for all features
        for letter in string.ascii_lowercase:
            if letter in guessed_letters:
                continue

            # N-gram frequencies, n: Length of n-gram
            considered_positions = [i for i in unknown_positions if (i - 1 >= 0 and current_state[i-1] != '_') 
                                    or (i + 1 < state_len and current_state[i+1] != '_')]
            for n, w in enumerate(self.w_ng, start=1):
                for i in considered_positions:
                    for j in range(n): 
                        if i-j >= 0 and i-j+n < state_len:  # Check if indices are valid
                            if current_state[i-j:i-j+n].count('_') >= 4:
                                continue
                            test_pattern = (current_state[:i] + letter + current_state[i+1:])[i-j:i-j+n]
                            unknown_test_positions = [k for k, c in enumerate(test_pattern) if c == '_']
                            replacement_candidates = list(set(string.ascii_lowercase) - set(guessed_letters))
                            for replacement in product(replacement_candidates, repeat=len(unknown_test_positions)):
                                char_list = list(test_pattern)
                                for pos, char in zip(unknown_test_positions, replacement):
                                    char_list[pos] = char
                                filled_test_pattern = ''.join(char_list)
                                if filled_test_pattern in self.fe.ngram_freqs[n]:
                                    letter_frequencies['ngram'][n][letter] += self.fe.ngram_freqs[n][filled_test_pattern]

            # Position probabilities
            if state_len >= 5:
                for i in unknown_positions:
                    quintile = min(4, int(i * 5 / state_len))
                    letter_frequencies['position'][letter] += self.fe.positional_quintile_freqs[quintile].get(letter, 0)

            for prefix_length, w in enumerate(self.w_pf, start=2):
                if state_len >= prefix_length + self.fe.additional_length:
                    prefix = current_state[:prefix_length]
                    for i in unknown_positions:
                        if i < prefix_length:
                            prefix_replaced = (prefix[:i] + letter + prefix[i+1:]).replace('_', '.')
                            if (prefix_replaced.count('.') >= 1 and prefix_length <= 3) or (prefix_replaced.count('.') >= 2 and prefix_length <= 5):
                                continue
                            compiled_prefix = re.compile(prefix_replaced)
                            for key_prefix, freq in self.fe.affix_freqs['prefix'][prefix_length].items():
                                if compiled_prefix.match(key_prefix):
                                    letter_frequencies['prefix'][prefix_length][letter] += freq

            for suffix_length, w in enumerate(self.w_sf, start=2):
                if state_len >= suffix_length + self.fe.additional_length:
                    suffix = current_state[-suffix_length:]
                    for i in unknown_positions:
                        if i >= state_len - suffix_length:
                            suffix_replaced = (suffix[:i - (state_len - suffix_length)] + letter + suffix[i - (state_len - suffix_length) + 1:]).replace('_', '.')
                            if (suffix_replaced.count('.') >= 1 and suffix_length <= 3) or (suffix_replaced.count('.') >= 2 and suffix_length <= 5):
                                continue
                            compiled_suffix = re.compile(suffix_replaced)
                            for key_suffix, freq in self.fe.affix_freqs['suffix'][suffix_length].items():
                                if compiled_suffix.match(key_suffix):
                                    letter_frequencies['suffix'][suffix_length][letter] += freq

            # Calculate current vowel-consonant ratio
            current_vowels = sum(1 for c in current_state if c != '_' and c in self.vowels)
            
            # Get target ratio from feature extractor for this word length
            current_ratio = current_vowels / number_of_knowns if number_of_knowns > 0 else 0
            target_ratio = self.fe.vowel_cons_stats['mean_ratio']
            
            # Adjust probabilities based on vowel/consonant needs, if the difference is significant
            if abs(current_ratio - target_ratio) > 2 * self.fe.vowel_cons_stats['std_ratio'] and number_of_knowns >= 3:
                target_vowels = int(state_len * target_ratio)
                needed_vowels = target_vowels - current_vowels
                
                # Adjust probabilities based on vowel/consonant needs
                vowel_adjustment = needed_vowels / number_of_unknowns
                consonant_adjustment = 1 - vowel_adjustment

                if letter in self.vowels:
                    letter_frequencies['vowel_ratio'][letter] = vowel_adjustment
                else:
                    letter_frequencies['vowel_ratio'][letter] = consonant_adjustment
            
        letter_probabilities = defaultdict(dict)
        for n in range(1, self.fe.max_ngram + 1):
            letter_probabilities['ngram'][n] = defaultdict(float)
        
        for n in range(2, self.fe.max_affix_length + 1):
            letter_probabilities['prefix'][n] = defaultdict(float)
            letter_probabilities['suffix'][n] = defaultdict(float)

        letter_probabilities['base'] = normalize_distribution(letter_frequencies['base'])
        for n in range(1, self.fe.max_ngram + 1):
            letter_probabilities['ngram'][n] = normalize_distribution(letter_frequencies['ngram'][n])
        letter_probabilities['position'] = normalize_distribution(letter_frequencies['position'])
        for n in range(2, len(self.w_pf) + 2):
            letter_probabilities['prefix'][n] = normalize_distribution(letter_frequencies['prefix'][n])
            letter_probabilities['suffix'][n] = normalize_distribution(letter_frequencies['suffix'][n])
        letter_probabilities['vowel_ratio'] = letter_frequencies['vowel_ratio']
        
        final_probs = defaultdict(float)
        for letter in string.ascii_lowercase:
            if letter in guessed_letters:
                continue
            weighted_base = self.w_base * letter_probabilities['base'].get(letter, 0)
            weighted_ngram = sum(w * letter_probabilities['ngram'][n].get(letter, 0) for n, w in enumerate(self.w_ng, start=1))
            weighted_position = self.w_pos * letter_probabilities['position'].get(letter, 0)
            weighted_prefix = sum(w * letter_probabilities['prefix'][n].get(letter, 0) for n, w in enumerate(self.w_pf, start=2))
            weighted_suffix = sum(w * letter_probabilities['suffix'][n].get(letter, 0) for n, w in enumerate(self.w_sf, start=2))
            weighted_vowel = self.w_vowel * letter_probabilities['vowel_ratio'].get(letter, 0)

            final_probs[letter] = weighted_base + weighted_ngram + weighted_position + weighted_prefix + weighted_suffix + weighted_vowel
        normalized_probs = normalize_distribution(final_probs)

        if self.debug:
            print('Current state:', current_state,
                    'Guessed letters:', guessed_letters)
            for letter in string.ascii_lowercase:
                if letter in guessed_letters:
                    continue
                print('Letter:', letter,
                'Base:', f"{letter_probabilities['base'].get(letter, 0):.3f}",
                '1ng:', f"{letter_probabilities['ngram'][1].get(letter, 0):.3f}",
                '2ng:', f"{letter_probabilities['ngram'][2].get(letter, 0):.3f}",
                '3ng:', f"{letter_probabilities['ngram'][3].get(letter, 0):.3f}",
                '4ng:', f"{letter_probabilities['ngram'][4].get(letter, 0):.3f}",
                '5ng:', f"{letter_probabilities['ngram'][5].get(letter, 0):.3f}",
                'Pos:', f"{letter_probabilities['position'].get(letter, 0):.3f}",
                'Pr2:', f"{letter_probabilities['prefix'][2].get(letter, 0):.3f}",
                'Su2:', f"{letter_probabilities['suffix'][2].get(letter, 0):.3f}",
                'Pr3:', f"{letter_probabilities['prefix'][3].get(letter, 0):.3f}",
                'Su3:', f"{letter_probabilities['suffix'][3].get(letter, 0):.3f}",
                'Pr4:', f"{letter_probabilities['prefix'][4].get(letter, 0):.3f}",
                'Su4', f"{letter_probabilities['suffix'][4].get(letter, 0)::.3f}",
                'Vow:', f"{letter_probabilities['vowel_ratio'].get(letter, 0):.3f}",
                'Fin:', f"{normalized_probs[letter]:.3f}"
                )
        
        # Special rules for high predictions from 4,5-grams and 4 length affixes, atomizing the probabilities 
        if self.atomize:
            for letter in normalized_probs:
                if (letter_probabilities['ngram'][4].get(letter, 0) > 0.85
                    or letter_probabilities['ngram'][5].get(letter, 0) > 0.85
                    or letter_probabilities['prefix'][4].get(letter, 0) > 0.85
                    or letter_probabilities['suffix'][4].get(letter, 0) > 0.85):
                    normalized_probs[letter] = 1.0

        return normalized_probs

    def select_next_letter(self, current_state: str, guessed_letters: Set[str]) -> str:
        """Select the next letter to guess based on current game state.

        Args:
            current_state: String with '_' for unknown letters.
            guessed_letters: Set of letters already guessed.

        Returns:
            The next letter to guess.
        """
        matching_words = self.fe.get_matching_words(current_state, guessed_letters)
        letter_probs = self._get_letter_probs(matching_words, current_state, guessed_letters)
        return max(letter_probs, key=letter_probs.get)

    def play_game(self, secret_word: str, max_incorrect: int = 6) -> Tuple[str, int, List[str], bool]:
        """Play a game of hangman with the given secret word.

        Args:
            secret_word: The word to guess.
            max_incorrect: Maximum number of incorrect guesses allowed.

        Returns:
            Tuple containing the final state, number of incorrect guesses, list of guessed letters, and win status.
        """
        current_state = "_"*len(secret_word)
        guessed_letters = list()
        incorrect = 0
        while "_" in current_state and incorrect < max_incorrect:
            guess = self.select_next_letter(current_state, set(guessed_letters))
            guessed_letters.append(guess)
            if guess in secret_word:
                current_state = ''.join([c if c == guess or current_state[i] != '_' else '_' for i, c in enumerate(secret_word)])
            else:
                incorrect += 1
            if self.debug:
                print(f"Secret word: {secret_word}, Guess: {guess}, Current state: {current_state}, Incorrect: {incorrect}")
        win = False
        if not '_' in current_state:
            win = True
        return current_state, incorrect, guessed_letters, win

def evaluate_predictor(word_list: np.ndarray,
                      solver: HangmanSolver,
                      trials: int = 100,
                      verbose: bool = False) -> float:
    """Evaluate the performance of a hangman solver.

    Args:
        word_list: Array of words to test against.
        solver: HangmanSolver instance to evaluate.
        trials: Number of games to play.
        verbose: Whether to print detailed results.

    Returns:
        Win rate as a float between 0 and 1.
    """
    solver.debug = False
    wins = 0
    p_bar = tqdm(range(trials))
    for trial in p_bar:
        w = np.random.choice(word_list)
        current_state, incorrect, guessed_letters, win = solver.play_game(w)
        if win:
            wins += 1
        p_bar.set_description(f"Win rate: {wins / (1 + trial):.3f}")
        if verbose:
            print(f"Word:{w}, Current state: {current_state}, Incorrect: {incorrect}, Guesses: {guessed_letters}, Win: {win}", 'Win Rate:', wins / (1 + trial))
    return wins / trials

words = np.loadtxt('words.txt', dtype=str)
train, test = train_test_split(words, test_size=0.2)

fe = FeatureExtractor(train)

solver = HangmanSolver(fe, w_base=0.3, w_pos=0.2,
                       w_1ng=0.6, w_2ng=0.5, w_3ng=0.6, w_4ng=0.5, w_5ng=0.6,
                       w_pf2=0.4, w_pf3=0.5, w_pf4=0.6, w_pf5=0.6,
                       w_sf2=0.5, w_sf3=0.6, w_sf4=0.1, w_sf5=0.5,
                       w_vowel=0.3)

def main() -> None:
    """Main function to play a single game of hangman."""
    word = np.random.choice(test)
    current_state, incorrect, sequential_guesses, win = solver.play_game(word)
    print(f"Word:{word}, Current state: {current_state}, Incorrect: {incorrect}, Guesses: {sequential_guesses}, Win: {win}")

class MyPruner(BasePruner):
    """Stops trials that are below 0.30 win rate after 60 trials or 0.50 win rate after 100 trials."""
    def prune(self, study: optuna.Study, trial: optuna.Trial) -> bool:
        if (trial.number >= 100 and trial.value < 0.50) or (trial.number >= 60 and trial.value < 0.30):
            return True
        return False

def optimize_solver_weights() -> None:
    """Optimize the weights for the hangman solver using Optuna."""
    import optuna
    from optuna.samplers import GridSampler
    import multiprocessing

    # Define starting point for hyperparameters
    starting_point = {
        'w_base': 0.2,
        'w_pos': 0.2,
        'w_1ng': 0.6,
        'w_2ng': 0.4,
        'w_3ng': 0.6,
        'w_4ng': 0.6,
        'w_5ng': 0.6,
        'w_pf2': 0.4,
        'w_pf3': 0.6,
        'w_pf4': 0.6,
        'w_pf5': 0.6,
        'w_sf2': 0.6,
        'w_sf3': 0.6,
        'w_sf4': 0.2,
        'w_sf5': 0.4,
        'w_vowel': 0.4
    }

    # Define perturbation amount
    perturbation = 0.1

    # Create parameter grid for GridSampler
    param_grid = {
        key: [
            max(0.0, starting_point[key] - perturbation),  # Perturb down
            starting_point[key],                           # Starting point
            min(1.0, starting_point[key] + perturbation)   # Perturb up
        ]
        for key in starting_point
    }

    # Create a GridSampler with the parameter grid
    sampler = GridSampler(param_grid)

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            Win rate as a float between 0 and 1.
        """
        # Suggest hyperparameters from the grid in 0.1 increments around the starting point
        params = {key: trial.suggest_float(key, 0.0, 1.0) for key in starting_point}

        # Create a new HangmanSolver instance with suggested hyperparameters
        solver = HangmanSolver(
            fe,
            w_base=params['w_base'],
            w_pos=params['w_pos'],
            w_1ng=params['w_1ng'],
            w_2ng=params['w_2ng'],
            w_3ng=params['w_3ng'],
            w_4ng=params['w_4ng'],
            w_5ng=params['w_5ng'],
            w_pf2=params['w_pf2'],
            w_pf3=params['w_pf3'],
            w_pf4=params['w_pf4'],
            w_pf5=params['w_pf5'],
            w_sf2=params['w_sf2'],
            w_sf3=params['w_sf3'],
            w_sf4=params['w_sf4'],
            w_sf5=params['w_sf5'],
            w_vowel=params['w_vowel']
        )

        # Evaluate the solver performance
        win_rate = evaluate_predictor(test, solver, trials=250, verbose=False)
        return win_rate  # Minimize negative win rate to maximize win rate

    # Create a study with the GridSampler
    storage = "sqlite:///example.db"
    # Check if study already exists, else create a new one
    if not optuna.study.get_all_study_summaries(storage):
        study = optuna.create_study(study_name='hangman_study',
                                    sampler=sampler,
                                    storage=storage,
                                    pruner=MyPruner(),
                                    direction='maximize'
                                    )
    else:
        study = optuna.load_study(study_name='hangman_study',
                                storage=storage,
                                sampler=sampler,
                                pruner=MyPruner(),
                                )

    # Optimize the objective function using all CPU cores
    study.optimize(objective, n_jobs=4)

    print("Best parameters:", study.best_params)
    print("Best win rate:", -study.best_value)

if __name__ == "__main__":
    evaluate_predictor(test, solver, trials=400, verbose=False)

main()