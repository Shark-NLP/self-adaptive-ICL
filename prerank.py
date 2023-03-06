import json
import logging
from collections import defaultdict

import faiss
import hydra
import hydra.utils as hu
import numpy as np
import random
import torch
import tqdm
from sentence_transformers import SentenceTransformer
import os
import datetime


from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from dppy.finite_dpps import FiniteDPP

from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.dataset_readers.prerank_dsr import PrerankDatasetReader
from src.utils.dpp_map import fast_map_dpp

logger = logging.getLogger(__name__)


class PreRank:
    def __init__(self, cfg) -> None:
        self.cuda_device = cfg.cuda_device

        self.retriever_model = SentenceTransformer(cfg.retriever_model).to(
            self.cuda_device) if cfg.retriever_model != 'none' else None

        self.retriever_model.eval()

        self.dataset_reader = PrerankDatasetReader(task_name=cfg.dataset_reader.task_name,
                                                   field=cfg.dataset_reader.field,
                                                   dataset_path=cfg.dataset_reader.dataset_path,
                                                   dataset_split=cfg.dataset_reader.dataset_split,
                                                   tokenizer=self.retriever_model.tokenizer)


        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer,
                                            device=self.cuda_device)

        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader.dataset_split == "train"
        self.dpp_sampling = cfg.dpp_sampling
        self.scale_factor = cfg.scale_factor
        self.dpp_topk = cfg.dpp_topk
        self.mode = "cand_selection"
        self.method = cfg.method
        self.vote_k_idxs = None
        self.vote_k_k = cfg.vote_k_k

        self.index_reader = PrerankDatasetReader(task_name=cfg.index_reader.task_name,
                                                 field=cfg.index_reader.field,
                                                 dataset_path=cfg.index_reader.dataset_path,
                                                 dataset_split=cfg.index_reader.dataset_split,
                                                 tokenizer=self.retriever_model.tokenizer)

        self.index = self.create_index(cfg)

    def create_index(self, cfg):
        logger.info("building index...")
        starttime = datetime.datetime.now()
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.index_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(self.index_reader, batch_size=cfg.batch_size, collate_fn=co)

        index = faiss.IndexIDMap(faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(768)))
        res_list = self.forward(dataloader)

        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.stack([res['embed'] for res in res_list])
        if self.method == 'votek':
            self.vote_k_idxs = self.vote_k_select(embeddings=embed_list, select_num=self.num_candidates,
                                                  k=self.vote_k_k,overlap_threshold=1)
        index.add_with_ids(embed_list, id_list)
        cpu_index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(cpu_index, cfg.index_file)
        endtime = datetime.datetime.now()
        logger.info(f"end building index, size {len(self.index_reader)}, time: {(endtime-starttime).seconds} seconds")
        return index

    def forward(self, dataloader, **kwargs):
        res_list = []
        logger.info(f"Totoal number of batches: {len(dataloader)}")
        for i, entry in enumerate(dataloader):
            with torch.no_grad():
                if i % 500 == 0:
                    logger.info(f"finish {str(i)} batches")
                metadata = entry.pop("metadata")
                raw_text = self.retriever_model.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True)
                res = self.retriever_model.encode(raw_text, show_progress_bar=False, **kwargs)  # 只要把model换掉就行?
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def knn_search(self, entry, num_candidates=1, num_ice=1):
        embed = np.expand_dims(entry['embed'], axis=0)
        near_ids = self.index.search(embed, max(num_candidates, num_ice) + 1)[1][0].tolist()  # 检索相似的embed
        near_ids = near_ids[1:] if self.is_train else near_ids
        return near_ids[:num_ice], [[i] for i in near_ids[:num_candidates]]  # candidates格式有点怪，但是不影响

    def random_search(self, num_candidates=1, num_ice=1):  # 没用
        rand_ids = np.random.choice(list(range(len(self.index_reader))), size=num_candidates, replace=False).tolist()
        return rand_ids[:num_ice], [[i] for i in rand_ids[:num_candidates]]  # candidates格式有点怪，但是不影响

    def get_kernel(self, embed, candidates):
        near_reps = np.stack([self.index.index.reconstruct(i) for i in candidates], axis=0)
        # normalize first
        embed = embed / np.linalg.norm(embed)
        near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)

        rel_scores = np.matmul(embed, near_reps.T)[0]
        rel_scores = (rel_scores + 1) / 2
        # to balance relevance and diversity
        rel_scores = np.exp(rel_scores / (2 * self.scale_factor))

        sim_matrix = np.matmul(near_reps, near_reps.T)
        sim_matrix = (sim_matrix + 1) / 2
        # print((sim_matrix < 0).sum())
        # print((rel_scores < 0).sum())
        kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
        return near_reps, rel_scores, kernel_matrix

    def k_dpp_sampling(self, kernel_matrix, rel_scores, num_ice, num_candidates):
        ctxs_candidates_idx = [list(range(num_ice))]
        dpp_L = FiniteDPP('likelihood', **{'L': kernel_matrix})
        i = 0
        while len(ctxs_candidates_idx) < num_candidates:
            try:
                samples_ids = np.array(dpp_L.sample_exact_k_dpp(size=num_ice, random_state=i))
            except Exception as e:
                logger.info(e)
                i += 1
                if (i > 9999999):
                    raise RuntimeError('Endless loop')
                continue
            i += 1
            # ordered by relevance score
            samples_scores = np.array([rel_scores[i] for i in samples_ids])
            samples_ids = samples_ids[(-samples_scores).argsort()].tolist()

            if samples_ids not in ctxs_candidates_idx:
                assert len(samples_ids) == num_ice
                ctxs_candidates_idx.append(samples_ids)

        return ctxs_candidates_idx

    def dpp_search(self, entry, num_candidates=1, num_ice=1):
        candidates = self.knn_search(entry, num_ice=self.dpp_topk)[0]
        embed = np.expand_dims(entry['embed'], axis=0)
        near_reps, rel_scores, kernel_matrix = self.get_kernel(embed, candidates)

        if self.mode == "cand_selection":
            ctxs_candidates_idx = self.k_dpp_sampling(kernel_matrix=kernel_matrix, rel_scores=rel_scores,
                                                      num_ice=num_ice, num_candidates=num_candidates)
        else:
            # MAP inference and create reordering candidates
            map_results = fast_map_dpp(kernel_matrix, num_ice)
            map_results = sorted(map_results)
            ctxs_candidates_idx = [map_results]
            while len(ctxs_candidates_idx) < num_candidates:
                # ordered by sim score
                ctxs_idx = map_results.copy()
                np.random.shuffle(ctxs_idx)
                if ctxs_idx not in ctxs_candidates_idx:
                    ctxs_candidates_idx.append(ctxs_idx)

        ctxs_candidates = []
        for ctxs_idx in ctxs_candidates_idx[:num_candidates]:
            ctxs_candidates.append([candidates[i] for i in ctxs_idx])
        assert len(ctxs_candidates) == num_candidates

        return ctxs_candidates[0], ctxs_candidates

    def vote_k_select(self, embeddings=None, select_num=None, k=None, overlap_threshold=None, vote_file=None):
        n = len(embeddings)
        if vote_file is not None and os.path.isfile(vote_file):
            with open(vote_file) as f:
                vote_stat = json.load(f)
        else:
            # bar = tqdm(range(n), desc=f'vote {k} selection')
            vote_stat = defaultdict(list)

            for i in range(n):
                cur_emb = embeddings[i].reshape(1, -1)
                cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
                sorted_indices = np.argsort(cur_scores).tolist()[-k - 1:-1]
                for idx in sorted_indices:
                    if idx != i:
                        vote_stat[idx].append(i)
                # bar.update(1)
            if vote_file is not None:
                with open(vote_file, 'w') as f:
                    json.dump(vote_stat, f)
        votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
        j = 0
        selected_indices = []
        while len(selected_indices) < select_num and j < len(votes):
            candidate_set = set(votes[j][1])
            flag = True
            for pre in range(j):
                cur_set = set(votes[pre][1])
                if len(candidate_set.intersection(cur_set)) >= overlap_threshold * len(candidate_set):
                    flag = False
                    break
            if not flag:
                j += 1
                continue
            selected_indices.append(int(votes[j][0]))
            j += 1
        if len(selected_indices) < select_num:
            unselected_indices = []
            cur_num = len(selected_indices)
            for i in range(n):
                if not i in selected_indices:
                    unselected_indices.append(i)
            selected_indices += random.sample(unselected_indices, select_num - cur_num)
        return selected_indices

    def vote_k_search(self, num_candidates=100, num_ice=8):
        return self.vote_k_idxs[:num_ice], [[i] for i in self.vote_k_idxs[:num_candidates]]

    def search(self, entry):
        if self.method == "random":
            return self.random_search(num_candidates=self.num_candidates, num_ice=self.num_ice)
        elif self.method == "topk":
            return self.knn_search(entry, num_candidates=self.num_candidates, num_ice=self.num_ice)
        elif self.method == "dpp" or self.dpp_sampling:
            return self.dpp_search(entry, num_candidates=self.num_candidates, num_ice=self.num_ice)
        elif self.method == "votek":
            return self.vote_k_search(num_candidates=self.num_candidates, num_ice=self.num_ice)

    def find(self):
        res_list = self.forward(self.dataloader)
        data_list = []
        starttime = datetime.datetime.now()
        for entry in res_list:
            data = self.dataset_reader.dataset_wrapper[entry['metadata']['id']]
            ctxs, ctxs_candidates = self.search(entry)
            data['ctxs'] = ctxs
            data['ctxs_candidates'] = ctxs_candidates
            data_list.append(data)

        endtime = datetime.datetime.now()
        logger.info(f"retrieval time: {(endtime-starttime).seconds} seconds")
        with open(self.output_file, "w") as f:
            json.dump(data_list, f)


@hydra.main(config_path="configs", config_name="prerank")
def main(cfg):
    logger.info(cfg)
    if not cfg.overwrite:
        if os.path.exists(cfg.output_file):
            logger.info(f'{cfg.output_file} already exists,skip')
            return
    logger.info(cfg)
    dense_retriever = PreRank(cfg)
    random.seed(cfg.rand_seed)
    np.random.seed(cfg.rand_seed)
    dense_retriever.find()


if __name__ == "__main__":
    main()