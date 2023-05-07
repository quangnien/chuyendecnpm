# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import numpy as np

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

path_queries = "./query-text"

def load_query():
    with open(path_queries, 'r') as f:
        queries = f.read()

    # tách từng queryID_query
    docID_queries = queries.lower().replace("\n"," ").split("/")
    docID_queries.pop()

    # tách queryID và query
    queries = []
    for query in docID_queries:
        query = query.strip()
        index = query.find(" ")
        queries.append(query[index+1:])
    
    return queries

#______________________________________#

if __name__ == '__main__':

    # Đọc file
    path_docs = "./doc-text"

    # Mở file và đọc nội dung vào biến 'documents'
    with open(path_docs, 'r') as f:
        documents = f.read()

    # tách từng docID_doc 
    docID_documents = documents.lower().replace("\n"," ").split("/")
    docID_documents.pop()

    # tách docID và doc
    # lưu nội dung doc vào danh sách documents.
    documents = []
    for document in docID_documents:
        document = document.strip()
        index = document.find(" ")
        documents.append(document[index+1:])

    # Tạo file từ vựng và file văn bản từ file doc - text
    np.save('./documents.npy', documents, allow_pickle=True)
    
    print("Done!")