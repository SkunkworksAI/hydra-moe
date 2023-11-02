from sentence_transformers import SentenceTransformer

class BGE_Large:
    def __init__(self, max_short_query_len=256, multi_gpu=False, batch_size=32):
        self.model = SentenceTransformer('BAAI/bge-large-en')
        # self.instruction = "Represent this sentence for searching relevant passages: "
        if multi_gpu:
            self.pool = self.model.start_multi_process_pool()
            self.batch_size = batch_size


    def embed_text(self, texts, show_progress_bar=False):
        # Possibly add instruction in the future for s2p
        if self.pool:
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=show_progress_bar)
        else:
            return self.model.encode_multi_process(texts, pool=self.pool, batch_size=self.batch_size)
