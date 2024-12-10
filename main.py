import re
import nltk
nltk.download('punkt')  # Tải công cụ tokenization
nltk.download('stopwords')  # Tải stopwords để loại bỏ các từ không cần thiết
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer
import numpy as np
import time
start_time = time.process_time()



# Regex to remove HTML code
# def extract_text(data):
#     cleaned_text = re.findall(r'>\s*(.*?)\s*<', data)
#     return ' '.join(cleaned_text)
def extract_text(data):
    cleaned_text = re.findall(r'>\s*(.*?)\s*<', data)
    return ' '.join([re.sub(r'\s+', ' ', sentence).strip() for sentence in cleaned_text])

# Thêm bước thính TextRank nữa
def pagerank_from_similarity(similarity_matrix, alpha=0.85, max_iter=100, tol=1.0e-6):
    """
    Args:
        similarity_matrix: Ma trận tương đồng (numpy.ndarray), kích thước (N x N),
                           trong đó similarity_matrix[i][j] là độ tương đồng từ node j đến node i.
        alpha: Hệ số damping (thường là 0.85).
        max_iter: Số lần lặp tối đa.
        tol: Ngưỡng sai số hội tụ.

    Returns:
        ranks: Numpy array chứa PageRank scores cho từng node.
    """
    num_nodes = similarity_matrix.shape[0]

    # Chuẩn hóa ma trận tương đồng để tạo ma trận chuyển tiếp (transition matrix)
    column_sums = similarity_matrix.sum(axis=0)
    transition_matrix = np.divide(similarity_matrix, column_sums, where=column_sums != 0)

    # Khởi tạo PageRank ban đầu
    ranks = np.full(num_nodes, 1 / num_nodes)

    for _ in range(max_iter):
        # Tính toán PageRank mới
        new_ranks = (1 - alpha) / num_nodes + alpha * transition_matrix @ ranks

        # Kiểm tra hội tụ
        if np.linalg.norm(new_ranks - ranks, ord=1) < tol:
            return new_ranks

        ranks = new_ranks

    return ranks


def summariztion(filename):
    line_count = 0
    # Xử lý văn bản và Tokenization
    with open('./Data_DUC_2002/DUC_TEXT/train/' + filename, 'r') as file:
        text = file.read().replace(',', '')
        text = extract_text(text)
        # lines = file.readlines()  # Đọc tất cả các dòng
        # line_count = len(lines)    # Đếm số dòng
        # text = file.read().replace('Â', '').replace('â€”','').replace('\xa0',' ')

    sentences = sent_tokenize(text)  # Chia thành các câu
    stop_words = set(stopwords.words("english"))

    sentence_count=len(sentences)
    print(sentence_count)

    # Xử lý từng câu
    processed_sentences = []
    for sent in sentences:
        words = word_tokenize(sent)
        processed_sentence = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        processed_sentences.append(processed_sentence)

    # Tính tần suất từ (Word Frequency)
    word_frequencies = {}
    for sentence in processed_sentences:
        for word in sentence:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    # Chuẩn hóa tần suất từ
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] /= max_frequency

    # Tính điểm cho từng câu, Dựa trên tần suất từ, tính điểm cho mỗi câu
    sentence_scores = {}
    for i, sent in enumerate(sentences):
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if i not in sentence_scores:
                    sentence_scores[i] = word_frequencies[word]
                else:
                    sentence_scores[i] += word_frequencies[word]
    # print('sentence_scores')
    # print(sentence_scores)

    # Convert dictionary to similarity matrix
    num_sentences = len(sentences)
    # similarity_matrix = np.zeros((num_sentences, num_sentences))

    # -------------------------
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Giả sử sentences là danh sách các câu đã được token hóa và xử lý
    # Tạo từ điển từ và vector cho từng câu
    from sklearn.feature_extraction.text import CountVectorizer

    # Tạo CountVectorizer để chuyển đổi văn bản thành vectors
    vectorizer = CountVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences).toarray()

    # Tính toán ma trận tương đồng cosine
    similarity_matrix = cosine_similarity(sentence_vectors)

    # In ma trận tương đồng
    print("Cosine Similarity Matrix:")
    print(similarity_matrix)

    # -------------------------


    for i in range(num_sentences):
        for j in range(num_sentences):
            if i in sentence_scores and j in sentence_scores:
                # You might want to adjust this similarity calculation based on your needs
                similarity_matrix[i][j] = abs(sentence_scores[i] - sentence_scores[j])

    # Now pass the numpy array instead of the dictionary
    # Tính PageRank scores
    pagerank_scores = pagerank_from_similarity(similarity_matrix)
    print('pagerank_scores')
    print(pagerank_scores)

    # Tạo dictionary chứa điểm PageRank cho mỗi câu
    sentence_pagerank = {i: score for i, score in enumerate(pagerank_scores)}

    # Sắp xếp câu theo điểm PageRank và chọn 3 câu có điểm cao nhất
    summary_sentences = sorted(sentence_pagerank, key=sentence_pagerank.get, reverse=True)[:3]

    # Tạo tóm tắt từ các câu đã chọn
    summary = ' '.join([sentences[i] for i in sorted(summary_sentences)])
    print(summary)

    # exit(0)


    # Tóm tắt văn bản
    # Sắp xếp câu theo điểm số và chọn một số câu làm tóm tắt
    # import heapq
    # summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
    # summary = ' '.join([sentences[i] for i in summary_sentences])
    # print(summary)


    # Văn bản tóm tắt tự động và văn bản tham chiếu
    generated_summary = summary  # Tóm tắt từ code của bạn
    reference_summary = """..."""  # Nội dung từ file d069f_result

    # Tạo công cụ tính điểm ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Tính điểm ROUGE
    # ROUGE-1: Đánh giá mức độ trùng khớp của các từ đơn (unigrams) giữa bản tóm tắt tự động và bản tham chiếu.
    # ROUGE-2: Đánh giá mức độ trùng khớp của các cặp từ liên tiếp (bigrams).
    # ROUGE-L: Đánh giá mức độ trùng khớp của chuỗi con chung dài nhất (LCS) giữa hai văn bản.
    scores = scorer.score(reference_summary, generated_summary)
    print("ROUGE Scores:")
    print("ROUGE-1: ", scores['rouge1'])
    print("ROUGE-2: ", scores['rouge2'])
    print("ROUGE-L: ", scores['rougeL'])


    # Đọc nội dung từ file kết quả tham chiếu
    with open('./Data_DUC_2002/DUC_SUM/' + filename, 'r', encoding='utf-8') as f:
        reference_summary = f.read()

    # Thực hiện tính điểm ROUGE như trên
    scores = scorer.score(reference_summary, generated_summary)
    print("ROUGE Scores:")
    print("ROUGE-1: ", scores['rouge1'])
    print("ROUGE-2: ", scores['rouge2'])
    print("ROUGE-L: ", scores['rougeL'])
    print(f"ROUGE-L: precision={scores['rougeL'].precision:.4f}, recall={scores['rougeL'].recall:.4f}, fmeasure={scores['rougeL'].fmeasure:.4f}")



    end_time = time.process_time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time:.4f} seconds")
    print(filename, line_count)

summariztion('d061j')

