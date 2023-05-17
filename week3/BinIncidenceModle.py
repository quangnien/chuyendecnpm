# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import  numpy as np

class BIM():
    def __init__(self, vectors, vocabs, documents):
        self.vectors = vectors
        self.vocabs = vocabs
        self.documents = documents
        self.weights = None

    #get document
    def get_doc(self, id):
        return self.documents[id-1]

    #get vector of document
    def get_vector(self, id):
        return self.vectors[id-1]

    #get vocabularies
    def get_vocabs(self):
        return self.vocabs

    #get matrix document vector
    def get_vectors(self):
        return self.vectors

    #init weights
    def init_RSV_weights(self):
        N = self.documents.shape[0]
        n = np.sum(self.vectors, axis=0) #n=df

        #ci = log( (N-ni+0.5)/(ni+0.5) )
        self.weights = np.log((N-n+0.5)/(n+0.5))


    #ranking
    def ranking(self, query):
        index = np.where(query==1)#xi_Q =1

        #caculate RSVd = SUM(xi=qi=1] (ci)
        scores = np.sum(self.vectors[:,index[0]]*self.weights[index], axis=1)

        #return scores of docs DESC
        ranks = np.argsort(-scores)
        return ranks, scores[ranks]

    #caculate weight when it response
    def recompute_weights(self, relevant_doc, query_vec):
        N = self.documents.shape[0] #number docs
        # N_rel = |V| number docs have rank highest
        N_rel = relevant_doc.shape[0]
        # index of qi = 1
        qi_1 = np.where(query_vec == 1)
        # Number doc has xi in V
        n_vi = np.sum(self.vectors[relevant_doc][:, qi_1[0]], axis=0)
        # pi = (|Vi| + 0.5) / (|V| + 1)
        pi = (n_vi + 0.5) / (N_rel + 1)
        n = np.sum(self.vectors[:, qi_1[0]], axis=0)  # df
        # ri = (n - |Vi| + 0.5) / (N - |Vi| + 1)
        ri = (n - n_vi + 0.5) / (N - N_rel + 1)
        # Update weight ci = log[ (pi.(1-ri))/(ri.(1-pi)) ]
        self.weights[qi_1] = np.log(pi / (1 - pi)) - np.log(ri / (1 - ri))


    def answer(self, query):
        #create vertor for query
        query_vec = np.isin(self.vocabs, query.split()).astype(int)

        #pi = 0.5^S, ri = ni/N
        self.init_RSV_weights()
        #implement when it pi and ri no change value or maximun times
        N_rel = 5  #number doc have rank highest
        n_loop = 10 #maximun times loop
        epsilon = 0.00001#thresh loop d(c_new,c_old)
        while True:
            n_loop = n_loop-1
            ranks, _ = self.ranking(query_vec)
            weights_old = np.array(self.weights)
            self.recompute_weights(ranks[:N_rel], query_vec)#VR=V
            if n_loop ==0 or np.sqrt(np.sum((self.weights - weights_old)**2, axis =0)) <epsilon:
                return self.ranking(query_vec)



if __name__ == '__main__':
    bim = BIM(
        np.load("./matrixMark.npy", allow_pickle=True),
        np.load('./vocabs.npy',allow_pickle=True),
        np.load("./docs.npy",allow_pickle=True)
    )

    #query
    from Preprocessing import load_query
    queries = load_query()
    with open("./rlv-ass.txt",'w') as file:

        for i, query in enumerate(queries):
            file.write(str(i+1)+"\n")
            print(f"QUERY {i+1}:{query}\n")
            ranking, scores = bim.answer(query)
            for index, score in list(zip(ranking, scores))[:5]:

                file.write(str(index+1)+" ")
                print(f"\nDOCUMENT {index+1}: {bim.get_doc(index+1)}\nScore: {score}\n/")
            file.write("\n/\n")

    print("Complete!")
