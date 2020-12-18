# This file parses password texts into phonemes to be furtehr converted into language models.
# but language models does not work well.

dictionary_path = ".\data\ASVspoof2017.dict"
text_path = ".\data\PasswordTexts.txt"
output_path = ".\data\PasswordPhonemes.txt"

# read phoneme dictionary and text
f = open(dictionary_path, "r")
phoneme_dict_text = f.read()
f.close()
f = open(text_path, "r")
text = f.read()
f.close()

# convert string to dictionary
phoneme_dict = {}
for line in phoneme_dict_text.split("\n"):
    word = line.split(" ")[0]
    phonemes = " ".join(line.split(" ")[1:])
    phoneme_dict[word.lower()] = phonemes

# convert text to phonemes
# replace each word with corresponding phonemes in dictionary
result = ""
for line in text.split("\n"):
    result += " ".join([phoneme_dict[word.lower()] for word in line.split(" ")]) + "\n"

f = open(output_path, "w")
f.write(result)
f.close()