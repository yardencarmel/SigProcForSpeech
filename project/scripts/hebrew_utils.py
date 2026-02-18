#!/usr/bin/env python3
"""
Hebrew Text Utilities for F5-TTS

This module provides utilities for processing Hebrew text:
- Normalizing Hebrew text (removing niqqud/vowel marks)
- Building vocabulary from Hebrew text
- Character validation and mapping
- Optional transliteration to Latin characters

Hebrew Unicode range: U+0590 to U+05FF
"""

import re
import unicodedata
from typing import Set, Dict, List, Optional
from collections import Counter


# Hebrew Unicode ranges
HEBREW_LETTERS = range(0x05D0, 0x05EB)  # א-ת (Alef to Tav)
HEBREW_FINALS = [0x05DA, 0x05DD, 0x05DF, 0x05E3, 0x05E5]  # Final letters ך ם ן ף ץ
NIQQUD_RANGE = range(0x0591, 0x05C7)  # Cantillation marks and vowels


def is_hebrew_letter(char: str) -> bool:
    """Check if a character is a Hebrew letter (excluding niqqud)."""
    if len(char) != 1:
        return False
    code = ord(char)
    return code in HEBREW_LETTERS


def is_hebrew_char(char: str) -> bool:
    """Check if a character is any Hebrew character (including niqqud)."""
    if len(char) != 1:
        return False
    code = ord(char)
    return 0x0590 <= code <= 0x05FF


def remove_niqqud(text: str) -> str:
    """
    Remove Hebrew niqqud (vowel marks) from text.
    
    Niqqud are diacritical marks added to Hebrew to indicate vowels.
    For TTS, we typically want just the consonant letters.
    
    Args:
        text: Hebrew text potentially containing niqqud
        
    Returns:
        Text with niqqud removed
    """
    result = []
    for char in text:
        code = ord(char)
        # Skip niqqud (cantillation and vowel marks)
        if code not in NIQQUD_RANGE:
            result.append(char)
    return ''.join(result)


def normalize_punctuation(text: str) -> str:
    """
    Normalize punctuation marks for consistency.
    
    Converts various quote styles and special characters to standard ASCII.
    """
    replacements = {
        '"': '"',   # Hebrew geresh
        '"': '"',   # Right double quote
        '"': '"',   # Left double quote
        ''': "'",   # Right single quote
        ''': "'",   # Left single quote
        '״': '"',   # Hebrew gershayim
        '׳': "'",   # Hebrew geresh
        '–': '-',   # En dash
        '—': '-',   # Em dash
        '…': '...',  # Ellipsis
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize multiple spaces and strip leading/trailing whitespace."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_hebrew_text(text: str, keep_niqqud: bool = False) -> str:
    """
    Full normalization pipeline for Hebrew text.
    
    Args:
        text: Input Hebrew text
        keep_niqqud: If True, preserve niqqud marks
        
    Returns:
        Normalized text ready for tokenization
    """
    if not keep_niqqud:
        text = remove_niqqud(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)
    return text


def extract_unique_chars(texts: List[str]) -> Set[str]:
    """
    Extract all unique characters from a list of texts.
    
    Used to build the vocabulary file.
    """
    chars = set()
    for text in texts:
        chars.update(text)
    return chars


def build_hebrew_vocab(
    texts: List[str],
    include_english: bool = True,
    include_numbers: bool = True,
    additional_chars: Optional[str] = None
) -> List[str]:
    """
    Build vocabulary from Hebrew texts.
    
    The vocabulary includes:
    1. Space (must be index 0 for F5-TTS compatibility)
    2. All Hebrew letters found in texts
    3. Punctuation marks
    4. Optionally: English letters and numbers
    
    Args:
        texts: List of normalized texts to extract vocabulary from
        include_english: Include a-z, A-Z
        include_numbers: Include 0-9
        additional_chars: Additional characters to include
        
    Returns:
        List of vocabulary characters (space first)
    """
    # Count character frequencies for better ordering
    char_counts = Counter()
    for text in texts:
        char_counts.update(text)
    
    # Start with space (MUST be index 0 for F5-TTS)
    vocab = [' ']
    
    # Hebrew letters (sorted by Unicode order)
    hebrew_chars = sorted([c for c in char_counts if is_hebrew_letter(c)])
    vocab.extend(hebrew_chars)
    
    # Common punctuation
    punctuation = ['.', ',', '!', '?', ':', ';', '-', "'", '"', '(', ')', '[', ']']
    for p in punctuation:
        if p not in vocab:
            vocab.append(p)
    
    # English letters if requested
    if include_english:
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if c not in vocab:
                vocab.append(c)
    
    # Numbers if requested
    if include_numbers:
        for c in '0123456789':
            if c not in vocab:
                vocab.append(c)
    
    # Additional characters
    if additional_chars:
        for c in additional_chars:
            if c not in vocab:
                vocab.append(c)
    
    # Any remaining characters from texts (excluding already added)
    remaining = sorted([c for c in char_counts if c not in vocab and c.isprintable()])
    vocab.extend(remaining)
    
    return vocab


def save_vocab(vocab: List[str], filepath: str) -> None:
    """
    Save vocabulary to file in F5-TTS format.
    
    Each character is on its own line.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for char in vocab:
            f.write(char + '\n')
    print(f"Saved vocabulary with {len(vocab)} characters to {filepath}")


def load_vocab(filepath: str) -> Dict[str, int]:
    """
    Load vocabulary from file.
    
    Returns:
        Dictionary mapping characters to indices
    """
    vocab_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Remove newline, keeping the character
            char = line[:-1] if line.endswith('\n') else line
            vocab_map[char] = i
    return vocab_map


# ============================================================================
# Phonetic Transliteration (Romanization)
# ============================================================================
# Improved Hebrew-to-Latin transliteration with:
# - Common word dictionary for accurate pronunciation
# - Mater lectionis handling (vav/yod as vowels)
# - Context-aware letter mapping

# Common Hebrew words with their correct transliteration
# This handles words where simple letter mapping fails
HEBREW_COMMON_WORDS = {
    'שלום': 'shalom',
    'עולם': 'olam',
    'שלומך': 'shlomcha',
    'שלומכם': 'shlomchem',
    'מה': 'ma',
    'איך': 'eich',
    'אני': 'ani',
    'אתה': 'ata',
    'את': 'at',
    'הוא': 'hu',
    'היא': 'hi',
    'אנחנו': 'anachnu',
    'אתם': 'atem',
    'הם': 'hem',
    'הן': 'hen',
    'זה': 'ze',
    'זאת': 'zot',
    'כן': 'ken',
    'לא': 'lo',
    'יש': 'yesh',
    'אין': 'ein',
    'טוב': 'tov',
    'רע': 'ra',
    'גדול': 'gadol',
    'קטן': 'katan',
    'יום': 'yom',
    'לילה': 'layla',
    'בוקר': 'boker',
    'ערב': 'erev',
    'תודה': 'toda',
    'בבקשה': 'bevakasha',
    'סליחה': 'slicha',
    'ישראל': 'israel',
    'ירושלים': 'yerushalayim',
    'תל': 'tel',
    'אביב': 'aviv',
    'חיים': 'chaim',
    'אהבה': 'ahava',
    'משפחה': 'mishpacha',
    'ילד': 'yeled',
    'ילדה': 'yalda',
    'אבא': 'aba',
    'אמא': 'ima',
    'בית': 'bayit',
    'ספר': 'sefer',
    'עברית': 'ivrit',
    'שפה': 'safa',
    'מילה': 'mila',
    'דבר': 'davar',
    'עבודה': 'avoda',
    'חבר': 'chaver',
    'חברה': 'chevra',
}

# Base consonant mapping
HEBREW_CONSONANTS = {
    'א': '',      # Alef - usually silent (glottal stop)
    'ב': 'v',     # Vet (default without dagesh)
    'ג': 'g',     # Gimel
    'ד': 'd',     # Dalet
    'ה': 'h',     # He (often silent at word end)
    'ז': 'z',     # Zayin
    'ח': 'ch',    # Chet
    'ט': 't',     # Tet
    'י': 'y',     # Yod (consonant form)
    'כ': 'ch',    # Chaf (without dagesh)
    'ך': 'ch',    # Final Chaf
    'ל': 'l',     # Lamed
    'מ': 'm',     # Mem
    'ם': 'm',     # Final Mem
    'נ': 'n',     # Nun
    'ן': 'n',     # Final Nun
    'ס': 's',     # Samech
    'ע': '',      # Ayin - silent in modern Hebrew
    'פ': 'f',     # Fe (without dagesh)
    'ף': 'f',     # Final Fe
    'צ': 'ts',    # Tsadi
    'ץ': 'ts',    # Final Tsadi
    'ק': 'k',     # Qof
    'ר': 'r',     # Resh
    'ש': 'sh',    # Shin (default, could be 's' for sin)
    'ת': 't',     # Tav
}


def transliterate_hebrew(text: str) -> str:
    """
    Transliterate Hebrew text to Latin characters phonetically.
    
    This improved version:
    - Uses a dictionary of common words for accuracy
    - Handles vav (ו) as 'o' or 'u' when used as vowel (mater lectionis)
    - Handles yod (י) as 'i' when used as vowel
    - Infers basic vowels between consonants
    
    Args:
        text: Hebrew text (should be normalized first)
        
    Returns:
        Romanized text (e.g., "שלום עולם" -> "shalom olam")
    """
    # Split into words and transliterate each
    words = text.split()
    result_words = []
    
    for word in words:
        # Extract just Hebrew letters for dictionary lookup
        hebrew_only = ''.join(c for c in word if is_hebrew_letter(c))
        
        # Check common words dictionary first
        if hebrew_only in HEBREW_COMMON_WORDS:
            # Find prefix (punctuation before Hebrew)
            prefix = ""
            suffix = ""
            hebrew_start = -1
            hebrew_end = -1
            
            for i, c in enumerate(word):
                if is_hebrew_letter(c):
                    if hebrew_start == -1:
                        hebrew_start = i
                    hebrew_end = i
            
            if hebrew_start > 0:
                prefix = word[:hebrew_start]
            if hebrew_end < len(word) - 1:
                suffix = word[hebrew_end + 1:]
            
            result_words.append(prefix + HEBREW_COMMON_WORDS[hebrew_only] + suffix)
        else:
            # Apply heuristic transliteration
            result_words.append(_transliterate_word(word))
    
    return ' '.join(result_words)


def _transliterate_word(word: str) -> str:
    """
    Transliterate a single Hebrew word with vowel inference.
    
    Handles mater lectionis (vowel letters):
    - ו after consonant often = 'o' or 'u'
    - י after consonant often = 'i'
    """
    result = []
    chars = list(word)
    i = 0
    
    while i < len(chars):
        char = chars[i]
        prev_char = chars[i-1] if i > 0 else None
        next_char = chars[i+1] if i < len(chars) - 1 else None
        
        if not is_hebrew_letter(char):
            # Keep punctuation and spaces as-is
            result.append(char)
            i += 1
            continue
        
        # Handle vav (ו) - can be 'v', 'o', or 'u'
        if char == 'ו':
            if prev_char and is_hebrew_letter(prev_char) and prev_char != 'ו':
                # After a consonant, likely a vowel
                # 'o' is more common than 'u' in Hebrew
                result.append('o')
            elif prev_char == 'ו':
                # Double vav = 'v' sound
                result.append('v')
            else:
                # Word initial or other = 'v'
                result.append('v')
            i += 1
            continue
        
        # Handle yod (י) - can be 'y' or 'i'  
        if char == 'י':
            if prev_char and is_hebrew_letter(prev_char) and prev_char not in 'וי':
                # After consonant, likely vowel 'i'
                result.append('i')
            else:
                result.append('y')
            i += 1
            continue
        
        # Handle he (ה) at word end - often silent or indicates 'a' vowel
        if char == 'ה' and next_char is None:
            # Final he - often silent, but can indicate 'a'
            if prev_char and is_hebrew_letter(prev_char):
                result.append('a')
            i += 1
            continue
        
        # Handle ayin (ע) - needs vowel
        if char == 'ע':
            # Ayin is silent but often followed by 'a' or 'e' sound
            if not result or result[-1] in ' ':
                result.append('a')  # Word initial ayin
            # Otherwise silent
            i += 1
            continue
        
        # Handle alef (א)
        if char == 'א':
            if not result or result[-1] in ' ':
                # Word initial - add 'a' vowel
                result.append('a')
            # Otherwise silent
            i += 1
            continue
        
        # Standard consonant mapping
        if char in HEBREW_CONSONANTS:
            result.append(HEBREW_CONSONANTS[char])
        
        i += 1
    
    return ''.join(result)


def demo_hebrew_processing():
    """Demonstrate Hebrew text processing."""
    # Sample Hebrew text with niqqud
    sample = "שָׁלוֹם עוֹלָם! מַה שְּׁלוֹמְךָ?"
    
    print("Original text:", sample)
    print("Without niqqud:", remove_niqqud(sample))
    print("Normalized:", normalize_hebrew_text(sample))
    print("Transliterated:", transliterate_hebrew(normalize_hebrew_text(sample)))
    
    # Build vocab from sample
    texts = [normalize_hebrew_text(sample)]
    vocab = build_hebrew_vocab(texts)
    print(f"\nVocabulary ({len(vocab)} chars):", vocab[:20], "...")


if __name__ == "__main__":
    demo_hebrew_processing()
