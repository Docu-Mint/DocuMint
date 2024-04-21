import re

# Calculate number of syllables in docstring
def count_syllables(word):
    # Remove punctuation
    word = re.sub(r'[^a-zA-Z]', '', word)
    
    # Vowel count
    vowels = 'aeiouy'
    syllables = 0
    last_was_vowel = False
    for char in word:
        if char.lower() in vowels:
            if not last_was_vowel:
                syllables += 1
            last_was_vowel = True
        else:
            last_was_vowel = False
    
    # Adjust syllable count for words ending in 'e'
    if word.endswith(('e', 'es', 'ed')):
        syllables -= 1
    
    # Adjust syllable count for words with no vowels
    if syllables == 0:
        syllables = 1
    
    return syllables

# Calculate Flesch reading score
def flesch_reading_ease(text):
    sentences = text.count('.') + text.count('!') + text.count('?') + 1
    words = len(re.findall(r'\b\w+\b', text))
    syllables = sum(count_syllables(word) for word in text.split())
    
    # Calculate Flesch Reading Ease score
    score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    
    return score

def main():
    text = input("Enter the text to calculate Flesch Reading Ease score: ")
    score = flesch_reading_ease(text)
    print("Flesch Reading Ease score:", score)

if __name__ == "__main__":
    main()