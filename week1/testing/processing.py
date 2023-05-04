import numpy as np

# stop_words là một danh sách các từ ngữ được xem là phổ biến 
# và không mang lại nhiều ý nghĩa cho việc phân tích ngôn ngữ tự nhiên. 
# Các từ này thường được loại bỏ khỏi các tài liệu văn bản để tập trung 
# vào các từ quan trọng hơn, đó là những từ mang lại ý nghĩa và đóng góp nhiều cho việc phân tích.
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Tạo tệp văn bản
# with open("text.txt", "w") as f:
#     f.write("This is the first sentence.\n")
#     f.write("This is the second sentence.\n")
#     f.write("This is the third sentence.\n")

# Đường dẫn đến tệp văn bản
path_docs = "./testing/doc-text.txt"

# Mở file và đọc nội dung vào biến 'documents'
with open(path_docs, 'r') as f:
    documents = f.read()

# Tách từng đoạn văn
docID_documents = documents.lower().replace("\n"," ").split("/")
docID_documents.pop()

# Tách docID và doc
# .find(" ") sẽ tìm vị trí của ký tự space đầu tiên trong đoạn văn bản. 
# Kết quả sẽ là một số nguyên, thể hiện vị trí của ký tự space.
# document[index+1:] sẽ lấy phần sau ký tự space (bắt đầu từ vị trí index+1) 
# trong đoạn văn bản làm nội dung của văn bản.
documents = []
for document in docID_documents:
    document = document.strip()
    index = document.find(" ")
    documents.append(document[index+1:])
# documents
# 0: 'this is the first sentence'
# 1: 'this is the second sentence'
# 2: 'this is the third sentence'
# len(): 3

# Tạo danh sách từ
# " ".join(documents) sẽ nối các đoạn văn bản trong danh sách documents bằng dấu cách để tạo ra một đoạn văn bản lớn.
# .strip() sẽ loại bỏ các khoảng trắng thừa ở đầu và cuối của đoạn văn bản lớn.
# .split() sẽ tách đoạn văn bản thành một danh sách các từ, mỗi từ được tách bởi dấu cách.
# set() sẽ tạo một tập hợp các từ khác nhau trong danh sách các từ, loại bỏ các từ trùng lặp.
# list() sẽ chuyển tập hợp các từ thành một danh sách.
vocabs = list(set(" ".join(documents).strip().split()))
# vocabs:
# 0: 'third'
# 1: 'first'
# 2: 'sentence'
# 3: 'this'
# 4: 'second'
# 5: 'the'
# 6: 'is'
# len(): 7
vocabs = [word for word in vocabs if word not in stop_words]


# Lưu danh sách văn bản vào tệp Numpy
np.save('./testing/documents.npy', documents, allow_pickle=True)

# Lưu danh sách từ vào tệp Numpy
np.save('./testing/vocabs.npy', vocabs, allow_pickle=True)

# In ra thông báo hoàn thành
print("Done!")
