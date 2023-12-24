import json
import os.path
import pickle




if __name__ == "__main__":
    # load embeddings
    with open("../data/retrieval_processed_embed_text/law_ndlr_embed_text_pairs.pkl", "rb") as f:
        retrieval_law_embed_pairs = pickle.load(f)
    with open("../data/retrieval_processed_embed_text/law_ndgr_embed_text_pairs.pkl", "rb") as f:
        retrieval_law_embed_pairs.extend(pickle.load(f))

    # load source information
    law_ndlr_sources = json.load(open("../data/retrieval_source_info/law_ndlr_source.json", "r"))
    law_ndgr_sources = json.load(open("../data/retrieval_source_info/law_ndgr_source.json", "r"))

    law_ndlr_sources.extend(law_ndgr_sources)

    # combine NDLR and NDGR rules
    law_comb_embeddings_path = "../data/retrieval_processed_embed_text/law_embed_text_pairs.pkl"
    law_comb_sources_path = "../data/retrieval_source_info/law_source.json"
    if not os.path.exists('/'.join(law_comb_embeddings_path.split('/')[:-1])):
        os.makedirs('/'.join(law_comb_embeddings_path.split('/')[:-1]))
    if not os.path.exists('/'.join(law_comb_sources_path.split('/')[:-1])):
        os.makedirs('/'.join(law_comb_sources_path.split('/')[:-1]))
    # store combined embeddings
    with open(law_comb_embeddings_path, "wb") as f:
        pickle.dump(retrieval_law_embed_pairs, f)
    # store combined sources
    json.dump(law_ndlr_sources, open(law_comb_sources_path, "w", encoding="utf8"), ensure_ascii=False)