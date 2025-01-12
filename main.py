# task1.py
class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(self.size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        key_hash = self.hash_function(key)
        key_value = [key, value]
        if self.table[key_hash] is None:
            self.table[key_hash] = list([key_value])
            return True
        else:
            for pair in self.table[key_hash]:
                if pair[0] == key:
                    pair[1] = value
                    return True
            self.table[key_hash].append(key_value)
            return True

    def get(self, key):
        key_hash = self.hash_function(key)
        if self.table[key_hash] is not None:
            for pair in self.table[key_hash]:
                if pair[0] == key:
                    return pair[1]
        return None

    def delete(self, key):
        key_hash = self.hash_function(key)
        if self.table[key_hash] is not None:
            for i, pair in enumerate(self.table[key_hash]):
                if pair[0] == key:
                    self.table[key_hash].pop(i)
                    return True
        return False


if __name__ == "__main__":
    H = HashTable(5)
    H.insert("apple", 10)
    H.insert("orange", 20)
    H.insert("banana", 30)

    print(H.get("apple"))   # Виведе: 10
    print(H.get("orange"))  # Виведе: 20
    print(H.get("banana"))  # Виведе: 30

    print(H.delete("apple"))   # Виведе: True
    print(H.get("apple"))      # Виведе: None
    print(H.delete("grape"))   # Виведе: False

# task2.py
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    iterations = 0
    upper_bound = float('inf')

    while left <= right:
        iterations += 1
        mid = (left + right) // 2

        if arr[mid] >= target and arr[mid] < upper_bound:
            upper_bound = arr[mid]

        if arr[mid] == target:
            return iterations, arr[mid]
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    if upper_bound == float('inf') and left < len(arr):
        upper_bound = arr[left]
    return iterations, upper_bound


if __name__ == "__main__":
    arr = [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9]
    test_values = [3.3, 5.0, 9.9, 10.0, 1.0]

    for target in test_values:
        iterations, upper_bound = binary_search(arr, target)
        print(f"\nSearching for {target}:")
        print(f"Number of iterations: {iterations}")
        print(f"Upper bound: {upper_bound}")

# task3.py
import timeit
from typing import List, Dict

def boyer_moore(text: str, pattern: str) -> int:
    def build_bad_match_table(pattern: str) -> Dict[str, int]:
        bad_match = {}
        length = len(pattern)
        for i in range(length - 1):
            bad_match[pattern[i]] = length - 1 - i
        return bad_match

    bad_match = build_bad_match_table(pattern)

    i = len(pattern) - 1
    while i < len(text):
        j = len(pattern) - 1
        k = i
        while j >= 0 and text[k] == pattern[j]:
            j -= 1
            k -= 1
        if j == -1:
            return k + 1
        else:
            i += bad_match.get(text[i], len(pattern))
    return -1

def kmp_search(text: str, pattern: str) -> int:
    def compute_lps(pattern: str) -> List[int]:
        lps = [0] * len(pattern)
        length = 0
        i = 1

        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = compute_lps(pattern)

    i = j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            return i - j
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1

def rabin_karp(text: str, pattern: str) -> int:
    d = 256
    q = 101

    def calculate_hash(string: str, length: int) -> int:
        result = 0
        for i in range(length):
            result = (d * result + ord(string[i])) % q
        return result

    pattern_length = len(pattern)
    text_length = len(text)

    if pattern_length > text_length:
        return -1

    pattern_hash = calculate_hash(pattern, pattern_length)
    text_hash = calculate_hash(text, pattern_length)

    h = pow(d, pattern_length - 1) % q

    for i in range(text_length - pattern_length + 1):
        if pattern_hash == text_hash:
            if text[i:i + pattern_length] == pattern:
                return i

        if i < text_length - pattern_length:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + pattern_length])) % q
            if text_hash < 0:
                text_hash += q

    return -1

def measure_search_time(func, text: str, pattern: str) -> float:
    search_func = lambda: func(text, pattern)
    execution_time = timeit.timeit(search_func, number=100) / 100
    return execution_time

def compare_algorithms(text1: str, text2: str, pattern_exists: str, pattern_not_exists: str) -> List[Dict]:
    algorithms = {
        "Boyer-Moore": boyer_moore,
        "KMP": kmp_search,
        "Rabin-Karp": rabin_karp
    }

    results = []

    for text_num, text in enumerate([text1, text2], 1):
        print(f"\nResults for Text {text_num}:")
        print("-" * 50)

        for pattern_type, pattern in [("Existing", pattern_exists),
                                      ("Non-existing", pattern_not_exists)]:
            print(f"\nPattern type: {pattern_type}")
            print(f"Pattern: '{pattern}'")
            print("-" * 30)

            for name, func in algorithms.items():
                time = measure_search_time(func, text, pattern)
                position = func(text, pattern)
                results.append({
                    "Text": f"Text {text_num}",
                    "Algorithm": name,
                    "Pattern Type": pattern_type,
                    "Time": time,
                    "Position": position
                })
                print(f"{name}:")
                print(f"  Time: {time:.6f} seconds")
                print(f"  Position: {position}")

    return results


if __name__ == "__main__":
    with open('article1.txt', 'r', encoding='latin1') as file:
        text1 = file.read()

    with open('article2.txt', 'r', encoding='latin1') as file:
        text2 = file.read()

    pattern_exists = "алгоритм"
    pattern_not_exists = "xyz123"

    results = compare_algorithms(text1, text2, pattern_exists, pattern_not_exists)