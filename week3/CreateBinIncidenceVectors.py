# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import numpy as np

from nltk.corpus import  stopwords
stop_words = stopwords.words('english')

# read file vocabularies and documents
vocabularies = np.load("./vocabs.npy",allow_pickle=True)
documents = np.load("./docs.npy",allow_pickle=True)

#create mark matrix
matrix = list(np.zeros((documents.shape[0], vocabularies.shape[0])))

for i in range(len(documents)):
    for w in documents[i].split():
        if w not in stop_words:
            matrix[i][np.where(vocabularies==w)[0][0]]=1


print(matrix)

#save mark matrix into file
np.save("./matrixMark.npy",matrix, allow_pickle=True)
print("Complete!")