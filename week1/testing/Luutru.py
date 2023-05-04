# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import numpy as np

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

vocabs = list(np.load("./testing/vocabs.npy", allow_pickle=True))
documents = list(np.load("./testing/documents.npy", allow_pickle=True))

# SINH THẺ ĐỊNH VỊ
# Lặp qua tất cả các tài liệu trong danh sách documents. 
# Với mỗi từ trong tài liệu, nếu từ đó không nằm trong danh sách stop_words, 
# chương trình tạo ra một thẻ định vị mới chứa từ đó và số hiệu tài liệu tương ứng.
term_docID = []
for i, doc in enumerate(documents):
    for word in doc.split():
        if word not in stop_words:
            term_docID.append([word, i + 1])

# XẾP THẺ ĐỊNH VỊ
# Các thẻ định vị này được sắp xếp theo từ và số hiệu tài liệu và lưu trữ trong biến term_docID.
term_docID= sorted(term_docID, key = lambda x: x[0])

# Tạo một từ điển trống dictionary. Với mỗi thẻ định vị trong term_docID, 
# nếu từ đó đã có trong từ điển, số hiệu tài liệu tương ứng sẽ được 
# thêm vào danh sách tài liệu đã có. Nếu từ chưa có trong từ điển, 
# một mục mới sẽ được thêm vào từ điển, với số hiệu tài liệu đầu tiên 
# được thêm vào danh sách tài liệu.
dictionary = {}
# Tổng hợp danh sách thẻ định vị
for row in term_docID:
    if row[0] in dictionary.keys():
        if row[1] not in dictionary[row[0]]:
            dictionary[row[0]].append(row[1])
    else:
        dictionary[row[0]] = [row[1]]

np.save("./invertedIndex.npy", dictionary, allow_pickle=True)

print("Done!")
