# N19DCCN056 - Phan Văn Hiểu
# N19DCCN096 - Cao Văn Lâm
# N19DCCN116 - Nguyễn Quang Niên

import processing
import numpy as np
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

class Inverted_Index:
    def __init__(self, path_dictionary, path_documents, skip = 1, optimal = False):
        # skip = 1: giao (intersect): duyệt tuần tự
        # skip > 1: giao + bước nhảy = skip (intersectWithSkips)
        # optimal = True: tối ưu
        self.dictionary = np.load(path_dictionary, allow_pickle=True).item()
        self.documents = np.load(path_documents, allow_pickle=True)

        # skip < 1 -> throw error
        if skip < 1:
            raise ValueError("skip >= 1") 
        self.skip = skip
        self.optimal = optimal

    
    # length() của posting-list
    # Số lượng tài liệu khác nhau trong hệ thống mà từ word xuất hiện trong chúng.
    # df là viết tắt của "document frequency" (tần suất xuất hiện trong tài liệu).
    # Ví dụ, nếu từ "apple" xuất hiện trong tài liệu 1 và 2, thì df("apple") sẽ trả về số 2.
    def df(self, word):
        return len(self.dictionary[word])

    # Trả về danh sách các tài liệu mà từ word xuất hiện trong chúng, 
    # được sắp xếp theo thứ tự tăng dần của số thứ tự tài liệu (docID). 
    def get_posting_list(self, word):
        return self.dictionary[word]   

    # lấy nội dung danh sách các tài liệu dựa trên danh sách các ID tài liệu cụ thể.
    # Hay : ấy nội dung danh sách các item trong cái posting-list
    # def get_documents(self, listDocID):
    #     results = []
    #     for docID in listDocID: 
    #         results.append([docID, self.documents[docID-1]])
    #     return results

    # Trả về giá trị của phần tử tại vị trí bước nhảy tiếp theo của danh sách posting list.
    def val_skip(self, p, i):
        return p[i + self.skip] 

    # Kiểm tra xem có bước nhảy tiếp theo hay không. 
    # Nếu vị trí hiện tại cộng với bước nhảy (self.skip) 
    # vượt quá độ dài của posting list thì không có bước nhảy tiếp theo 
    # và phương thức trả về giá trị False. Nếu không, phương thức trả về giá trị True.
    def hasSkip(self, p, i):
        return False if i + self.skip >= len(p) else True
    
    # tìm kiếm các từ khóa xuất hiện đồng thời trong hai danh sách các tài liệu.
    def intersect_2_set(self, p1, p2):
        return self.intersect(p1, p2) if self.skip == 1 else self.intersectWithSkips(p1, p2)
    
    # lấy ra những văn bản đồng thời chứa từ word này
    # p1, p2: 2 cái posting-list mình so sánh
    # return về tập giao của 2 posting-list
    def intersect(self, p1, p2):
        i = 0
        j = 0
        answer = []
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                answer.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j +=1       

        return answer    

    def intersectWithSkips(self, p1, p2):
        i = 0
        j = 0
        answer = []
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                answer.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                if self.hasSkip(p1, i) and self.val_skip(p1, i) <= p2[j]:
                    while self.hasSkip(p1, i) and self.val_skip(p1, i) <= p2[j]:
                        i += self.skip
                else:
                    i += 1
            else:
                if self.hasSkip(p2, j) and self.val_skip(p2, j) <= p1[i]:
                    while self.hasSkip(p2, j) and self.val_skip(p2, j) <= p1[i]:
                        j += self.skip
                else:
                    j += 1    
        return answer
    
    # tối ưu câu truy vấn
    def optimize(self, tokens): 
        # xếp theo thứ tự tăng dần df
        # các từ khóa được sắp xếp theo thứ tự tăng dần của độ phổ biến của chúng, 
        # các từ có độ phổ biến thấp hơn sẽ được sử dụng trước, 
        # từ đó giúp giảm thời gian tìm kiếm và cải thiện kết quả truy vấn.

        # sắp xếp tăng dần dựa trên df 
        return sorted(tokens, key = lambda word: self.df(word))  
        # tokens: T1,T2,.... term. vd: computer
    
    def query(self, str_query):

        # tách các thành phần trong câu query 
        tokens = str_query.lower().split()
        tokens = [word for word in tokens if word not in stop_words]

        # Thực hiện câu query      
        try:  
            if self.optimal == True:
                tokens = self.optimize(tokens)
            listDocID = self.get_posting_list(tokens.pop(0))
            while len(tokens) != 0:
                listDocID = self.intersect_2_set(listDocID , self.get_posting_list(tokens.pop(0)))
            return listDocID   
        except KeyError:
            return []

if __name__ == '__main__':


    I_index = Inverted_Index("./invertedIndex.npy", "./documents.npy", skip = 3, optimal= True)

    queries = processing.load_query()

    with open("rlv-ass", "w") as f:
        for j, query in enumerate(queries):
            print("Query", j +1, ":", query)
            listDocID = I_index.query(query)
            f.write(f"{j+1}\n")
            listDocID = I_index.query(query)
            for i, docID in enumerate(listDocID):
                f.write(f"{docID} ")
                if (i+1) % 10 == 0:
                    f.write("\n ")
            f.write("\n   /\n")