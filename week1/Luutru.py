# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import numpy as np

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

documents = list(np.load("./documents.npy", allow_pickle=True))

# SINH THẺ ĐỊNH VỊ
term_docID = []
for i, doc in enumerate(documents):
    for word in doc.split():
        if word not in stop_words:
            term_docID.append([word, i + 1])

# XẾP THẺ ĐỊNH VỊ
term_docID= sorted(term_docID, key = lambda x: x[0])

# Tạo một từ điển trống dictionary.
dictionary = {}
for row in term_docID:
    if row[0] in dictionary.keys():
        if row[1] not in dictionary[row[0]]:
            dictionary[row[0]].append(row[1])
    else:
        dictionary[row[0]] = [row[1]]

np.save("./invertedIndex.npy", dictionary, allow_pickle=True)

print("Done!")