# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import numpy as np


def import_data(path_file):
    '''
    This function import all articles in doc-tex,
    returning list of lists where each sub-list contains all the
    terms present in the document as a string
    '''

    with open(path_file,'r') as file:
        articles = file.read()

    #seperate docID_doc
    docID_documents = articles.lower().replace("\n"," ").split("/")
    docID_documents.pop()

    #seperate docID and document
    documents = []
    for doc in docID_documents:
        doc = doc.strip()
        index = doc.find(" ")
        documents.append(doc[index+1:])

    print(documents)

    #create list vocabulary
    vocabularies = list(set(" ".join(documents).strip().split()))
    vocabularies = [w for w in vocabularies if w not in stop_words]

    print(vocabularies)

    #save documents and vacabularys into file
    np.save("./docs.npy", documents, allow_pickle=True)
    np.save("./vocabs.npy", vocabularies, allow_pickle=True)


def load_query():
    with open('./query-text','r') as file:
        queries = file.read()


    #seperate queryID_queries
    queryID_queries = queries.lower().replace("\n"," ").split("/")
    queryID_queries.pop()

    #seperate queryID and queyy
    queries = []
    for query in queryID_queries:
        query = query.strip()
        index = query.find(" ")
        queries.append(query[index+1:])

    return queries
if __name__ == '__main__':
    import_data('./doc-text')
    print("Complete!")