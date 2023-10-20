import re


# melakukan ekstraksi kalimat yang berada di dalam bracket
def extract_bracketed_sentences(text, target_sentence, target_index):
    bracketed_sentences = re.findall(r"\[(.*?)\]", text)
    matching_sentences = []

    for sentence in bracketed_sentences:
        if (
            target_sentence in sentence
            and sentence.index(target_sentence) <= target_index
        ):
            matching_sentences.append(sentence)

    return matching_sentences


# memisahkan bracket satu dengan yang lainnya
def separate_brackets(string):
    new_string = ""
    for char in string:
        if char in ["[", "]"]:
            new_string += " " + char + " "
        else:
            new_string += char
    return new_string
