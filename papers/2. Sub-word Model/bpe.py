import re, collections

def get_stats(vocab):
    # vocab를 character 단위로 분리하여 frequency를 count
    pairs = collections.defaultdict(int)

    for word, freq in vocab.items():
        # 각 unit은 띄어쓰기로 구분되어 있기 때문에, 이러면 unit끼리 비교가 가능
        symbols = word.split()

        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
        
    return pairs

def merge_vocab(pair, v_in):
    # vocab v_in에 대해 pair의 공백을 제거하고 merge
    v_out = {}
    # re.escpae: 특수문자를 escape -> 왜?
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    # r'(?<!\S)': Negative Lookbehind : \S이 일치하지 않고 bigram이 나올 경우 반환
    # r'(?!\S)': Negative Lookahead :  bigram 다음 \S이 일치하지 않을 경우 반환
    # 즉, 공백(\S이 아님) + bigram(e\\ s) + 공백 제외(\S)가 나오는 형태를 의미
    # 다음과 동일
    # for word, freq in dictionary.items():
        # paired = word.replace(" ".join(pair), "".join(pair))
        # result[paired] = dictionary[word]

    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    
    return v_out

vocab = {'l o w </w>' : 5,
         'l o w e r </w>' : 2,
         'n e w e s t </w>':6,
         'w i d e s t e s t </w>':3
         }

num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get) # max value's key
    vocab = merge_vocab(best, vocab)
    print(vocab)
    