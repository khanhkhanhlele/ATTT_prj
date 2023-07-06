from math import log

def TF(doc):
    tf = {}
    wordDict = {}
    for sentence in doc:
        for word in sentence:
            if word not in wordDict.keys():
                wordDict[word] = 1
            else:
                wordDict[word] += 1
    numWord = sum(wordDict.values())
    for word in wordDict.keys():
        tf[word] = wordDict[word]/numWord
    return tf
  
def IDF(doc):
    idf = {}
    wordDict = {}
    for sentence in doc:
        tmp = []
        for word in sentence:
            if word not in wordDict.keys():
                wordDict[word] = 1
                tmp.append(word)
            else:
                if word not in tmp:
                    wordDict[word] += 1
                    tmp.append(word)
    docSize = len(doc)
    for word in wordDict.keys():
        idf[word] = wordDict[word]/docSize
    return idf
  
def TF_IDF(doc):
    tf_idf = {}
    tf = TF(doc)
    idf = IDF(doc)
    for word in tf.keys():
        tf_idf[word] = log(tf[word]*idf[word])
    return tf_idf
