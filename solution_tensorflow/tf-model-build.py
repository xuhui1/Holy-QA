
from utils.squad_preprocess import context_question_text_preprocess

embedding_matrix, vocab_size, pad_txt_train_cnt, pad_txt_train_qst, pad_txt_dev_cnt, pad_txt_dev_qst, \
    idx_train_beg, idx_dev_beg = context_question_text_preprocess()