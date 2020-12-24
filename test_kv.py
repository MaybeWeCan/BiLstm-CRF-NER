def extract_kvpairs_in_bio(bio_seq, word_seq):
    assert len(bio_seq) == len(word_seq)
    pairs = set()
    pre_bio = "O"
    v = ""
    for i, bio in enumerate(bio_seq):
        if (bio == "O"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = ""
        elif (bio[0] == "B"):
            if v != "": pairs.add((pre_bio[2:], v))
            v = word_seq[i]
        elif (bio[0] == "I"):
            if (pre_bio[0] == "O") or (pre_bio[2:] != bio[2:]):
                if v != "": pairs.add((pre_bio[2:], v))
                v = ""
            else:
                v += word_seq[i]
        pre_bio = bio
    if v != "": pairs.add((pre_bio[2:], v))
    return pairs


bio_seq = ["O","O","B","I","O","B","I","O"]
word_seq = ["我","爱","北","京","，","上","海","。"]

print(extract_kvpairs_in_bio(bio_seq, word_seq))

