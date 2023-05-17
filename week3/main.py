# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

from collections import  defaultdict
from math import log
import  re

from copy import deepcopy
from nltk.corpus import stopwords


def import_data(path_file):
    '''
    This function import all articles in doc-tex,
    returning list of lists where each sub-list contains all the
    terms present in the document as a string
    '''
    articles = []
    with open(path_file,'r') as file:
        listTemp = []
        for row in file:
            if row.endswith("/\n"):
                if listTemp != []:
                    articles.append(listTemp)
                listTemp=[]
            else:
                row = re.sub(r'[^a-zA-Z\s]+', '', row)
                listTemp += row.lower().split()
    return articles


def remove_stop_words(corpus):
    '''
    This function removes from the corpus all the stop words
    :param corpus:
    :return:
    '''
    stop_words = stopwords.words('english')
    for i in range(0,len(corpus)):
        corpus[i] = [x for x in corpus[i] if x not in stop_words]
    return corpus

def make_inverted_index(corpus):
    '''
    This function builds an inverted in dex as an hash table (dictionary) where the keyys are
    the terms and the values are ordered lists of docIDs containing the term
    :param corpus:
    :return:
    '''

    corpus = remove_stop_words(corpus)
    index = defaultdict(set)

    for docId, article in enumerate(corpus):
        for term in article:
            index[term].add(docId)
    return index


######################### Union Two posting litsts ########################################################

def posting_lists_union(pl1,pl2):
    '''
    Return new posting list resulting from the union of two lists passed as arguments
    :param pl1: posting litsts 1
    :param pl2: posting litsts 2
    :return:
    '''
    pl1 = sorted(list(pl1))
    pl2 = sorted(list(pl2))

    union = []
    i=0
    j=0

    while(i<len(pl1) and j<len(pl2)):
        if(pl1[i] == pl2[j]):
            union.append(pl1[i])
            i+=1
            j+=1
        elif (pl1[i] <pl2[j]):
            union.append(pl1[i])
            i+=1
        else:
            union.append(pl2[j])
            j+=1

    for k in range(i, len(pl1)):
        union.append(pl1[k])
    for k in range(j, len(pl2)):
        union.append(pl2[k])
    return union

################################# Precomputing weights ###############################################
def DF(term, index):
    '''
    Fuction computing Document Frequency for a term
    :param term:
    :param index:
    :return:
    '''
    return len(index[term])

def IDF(term, index, corous):
    '''
    Function computing Inverse Document Frequency for a term.
    :param term:
    :param index:
    :param corous:
    :return:
    '''

    return log(len(corous)/DF(term, index))

def RSV_weights(corpus, index):
    '''
    This function precomputes the Retrieval Status Value weights
    :param corpus:
    :param index:
    :return:
    '''
    N = len(corpus)
    w = {}
    for term in index.keys():
        p = DF(term, index)/(N+0.5)
        w[term] = IDF(term, index, corpus) + log(p/(1-p))
    return w

##################################   BIM Class  ################################################


class BIM():
    '''
    Binary Independence Model class
    '''

    def __init__(self, corpus):
        self.original_corpus = deepcopy(corpus)
        self.articles = corpus
        self.index = make_inverted_index(self.articles)
        self.weights = RSV_weights(self.articles, self.index)
        self.ranked = []
        self.query_text = ''
        self.N_retrieved = 0


    def RSV_doc_query(self, doc_id, query):
        '''
        This function computes the Retrieval Status Value for a given couple document - query
        using the precomputed weights
        :param doc_id:
        :param query:
        :return:
        '''
        score = 0
        doc = self.articles[doc_id]
        for term in doc:
            if term in query:
                score+=self.weights[term]
        return score

    def ranking(selt, query):
        '''
        Computes the score only for documents that are in the posting list of least one term in the query
        :param query:
        :return:
        '''

        docs = []
        for term in selt.index:
            if term in query:
                docs = posting_lists_union(docs, selt.index[term])

        scores = []
        for doc in docs:
            scores.append((doc, selt.RSV_doc_query(doc, query)))

        selt.ranked = sorted(scores, key= lambda  x : x[1], reverse=True)
        return selt.ranked


    def recompute_weights(self, relevant_idx, query):
        '''
        Suport relevance_feedback fuction and pseduo relevance feedback in answer_query.
        Recomputes the weights, only for the terms in the query based on a set of relevant documents
        :param relevant_idx:
        :param query:
        :return:
        '''

        relevant_docs = []
        for idx in relevant_idx:
            doc_id = self.ranked[idx-1][0]
            relevant_docs.append(self.articles[doc_id])

        N = len(self.articles)
        N_rel = len(relevant_idx)

        for term in query:
            if term in self.weights.keys():
                vri = 0
                for doc in relevant_docs:
                    if term in doc:
                        vri+=1
                p = (vri + 0.5)/(N_rel +1)
                u = (DF(term, self.index) - vri + 0.5) / (N-N_rel+1)
                self.weights[term] = log((1-u)/u)+log(p/(1-p))

    def answer_query(self, query_text):
        '''
        Function to answer a free text query.
        Implements the pseduo relevance feedback with k=5
        :param query_text:
        :return:
        '''

        # self.query_text = query_text
        query = query_text

        ranking = self.ranking(query)

        # pseudo relevance feedback
        i = 0
        new_ranking = []
        while i<10 and ranking !=new_ranking:
            self.recompute_weights([1,2,3,4,5], query)
            new_ranking = self.ranking(query)
            i+=1

        ranking = new_ranking
        self.ranked = ranking



        self.N_retrieved = 15

        #print retrieved documents


        for i in range(0, self.N_retrieved):
            article = self.original_corpus[ranking[i][0]]
            # if (len(article)>15):
            #     article = article[0:15]
            text = " ".join(article)
            print(f"Article {i+1}, docID: {ranking[i][0]} ,score: {ranking[i][1]}")
            print(text,'\n')

        self.weights = RSV_weights(self.articles, self.index)


def load_query():
    with open("./query-text", 'r') as file:
        queries = file.read()

    #seperate queryID_query
    docId_queries = queries.lower().replace("\n"," ").split("/")
    docId_queries.pop()


    #seperate queryId and query
    queries2 = []
    for query in docId_queries:
        query = query.strip()
        index = query.find(" ")
        queries2.append(query[index+1:])


    queries2 = remove_stop_words(queries2)
    return queries2


if __name__ == '__main__':
    #load articles from doc-text file
    articles = import_data('./doc-text')

    #init BIM for articles
    bim = BIM(articles)

    #load query from query-text file
    queries = import_data('./query-text')
    queries = remove_stop_words(queries)

    queryID=1  # idQuery
    f = open("result.txt", "w")

    for query in queries:
        QUERY = " ".join(query).upper()
        print("QUERY "+str(queryID)+": " + QUERY+"\n")

        # save ID query result into result.txt file
        f.write(str(queryID) + "\n")
        bim.answer_query(query)

        # save answer into result.txt file
        for i in range(0, bim.N_retrieved):
            f.write(str(bim.ranked[i][0]) + " ")

        f.write("\n\t/\n") #marked finish answer
        queryID = queryID + 1
    f.close()

    print('Complete!')