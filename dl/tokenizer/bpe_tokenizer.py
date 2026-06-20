
class BPE():
    def __init__(self):
        self.vocab = {}         # char -> id
        self.merges = {}
        self.id_to_char = {}
    
    def train(self, corpus, vocab_size):
        # 预处理, 按空格切词
        words = corpus.split(' ')   # 按空格切词
        chars = set()
        word_list = []
        for word in words:
            word = list(word)
            word.append('</w>')
            word_list.append(word) 
            for char in word:
                chars.add(char)
        for idx, char in enumerate(chars):
            self.vocab[char] = idx
            self.id_to_char[idx] = char
        word_ids = []
        for word in word_list:
            word_id =[]
            for char in word:
                word_id.append(self.vocab[char])
            word_ids.append(word_id)
        print(self.vocab)
        print(word_ids)


        cur_size = len(self.vocab)
        while cur_size < vocab_size:
            # BPE 迭代合并
            # 1. 统计所有词内相邻 pair 的频率
            freq_pair = {}
            for word in word_ids:
                for i in range(len(word) - 1):
                    freq_pair[word[i], word[i+1]] = freq_pair.get((word[i], word[i+1]), 0) + 1


            # 2. 找到频率最高的pair
            max_freq = 0
            max_first =  max_second = -1
            for (a, b), freq in freq_pair.items():
                if freq > max_freq:
                    max_first = a
                    max_second = b
                    max_freq = freq

            # 3. 合并：遍历 word_ids，所有 (a,b) 替换为 new_id
            for idx, word in enumerate(word_ids):
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and word[i]==max_first and word[i+1]==max_second:
                        new_word.append(cur_size)
                        i += 2   # 跳过两个
                    else:
                        new_word.append(word[i])
                        i += 1
                word_ids[idx] = new_word

            # 4. 记录 merges[(a,b)] = new_id
            self.merges[(max_first, max_second)] = cur_size
            
            # 5. 更新 vocab 和 id_to_char
            char = self.id_to_char[max_first] + self.id_to_char[max_second]
            
            self.vocab[char] =cur_size
            self.id_to_char[cur_size] = char
            cur_size += 1

            print(self.merges)

        print("Decoded:", bpe.decode(word_ids))
  


    

    def encode(self, text):
        words = text.split(' ')
        chars = set()
        word_list = []
        for word in words:
            word = list(word)
            word.append('</w>')
            word_list.append(word) 
            for char in word:
                chars.add(char)
        word_ids = []
        for word in word_list:
            word_id = []
            for char in word:
                word_id.append(self.vocab[char])
            word_ids.append(word_id)

        for pair, key in self.merges.items():
            for idx, word_id in enumerate(word_ids):
                new_id = []
                i = 0
                while i < len(word_id):
                    if i < len(word_id)-1 and word_id[i]==pair[0] and word_id[i+1]==pair[1]:
                        new_id.append(key)
                        i += 2   # 跳过两个
                    else:
                        new_id.append(word_id[i])
                        i += 1
                word_ids[idx] = new_id  

        return word_ids
        

    def  decode(self, ids):
        text = ""
        for id in ids:
            for ele in id:
                char =  self.id_to_char[ele]
                char = char.replace('</w>', ' ')
                text += char
        return text
        
        

bpe = BPE()
corpus = "low low low low low lower lower newest newest newest newest newest newest widest widest"
bpe.train(corpus,16)
print(bpe.encode("low newest"))
print(bpe.decode(bpe.encode("low newest")))