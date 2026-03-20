#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <algorithm>

namespace py = pybind11;

class MinBPEEncoder {
public:
    MinBPEEncoder(const std::vector<int32_t> &lefts,
                  const std::vector<int32_t> &rights,
                  const std::vector<int32_t> &ranks,
                  const std::vector<int32_t> &new_ids)
    {
        const size_t n = lefts.size();
        int32_t max_id = 0;
        for (size_t i = 0; i < n; ++i) {
            if (lefts[i] > max_id) max_id = lefts[i];
            if (rights[i] > max_id) max_id = rights[i];
        }
        // 每个 left 一个桶，右侧用 unordered_map（局部小表，查找更快）
        rank_tbl_.assign(static_cast<size_t>(max_id) + 1, {});
        newid_tbl_.assign(static_cast<size_t>(max_id) + 1, {});
        for (size_t i = 0; i < n; ++i) {
            auto L = static_cast<size_t>(lefts[i]);
            rank_tbl_[L][rights[i]] = ranks[i];
            newid_tbl_[L][rights[i]] = new_ids[i];
        }
    }

    std::vector<int32_t> encode(const std::vector<int32_t> &seq_in) const {
        return encode_with_limit(seq_in, INT32_C(0x7fffffff));
    }

    std::vector<int32_t> encode_with_limit(const std::vector<int32_t> &seq_in, int32_t rank_limit) const {
        if (seq_in.empty()) return {};
        std::vector<int32_t> seq = seq_in;            // 工作缓冲 A
        std::vector<int32_t> out; out.resize(seq.size()); // 工作缓冲 B（容量一次到位）

        while (true) {
            // 1) 找最小 rank 的 pair
            int32_t best_rank = INT32_C(0x7fffffff);
            int32_t best_l = -1, best_r = -1;
            const size_t n = seq.size();
            if (n < 2) break;
            for (size_t i = 0; i + 1 < n; ++i) {
                int32_t l = seq[i];
                int32_t r = seq[i + 1];
                if (l >= 0 && static_cast<size_t>(l) < rank_tbl_.size()) {
                    const auto &mp = rank_tbl_[static_cast<size_t>(l)];
                    auto it = mp.find(r);
                    if (it != mp.end()) {
                        int32_t rank = it->second;
                        if (rank >= rank_limit) continue;
                        if (rank < best_rank) {
                            best_rank = rank;
                            best_l = l;
                            best_r = r;
                        }
                    }
                }
            }
            if (best_rank == INT32_C(0x7fffffff)) {
                break;
            }

            // 2) 左到右执行该 pair 的合并
            size_t out_len = 0;
            for (size_t i = 0; i < n; ) {
                if (i + 1 < n && seq[i] == best_l && seq[i + 1] == best_r) {
                    const auto &mpn = newid_tbl_[static_cast<size_t>(best_l)];
                    auto itn = mpn.find(best_r);
                    int32_t new_id = itn->second;
                    out[out_len++] = new_id;
                    i += 2;
                } else {
                    out[out_len++] = seq[i];
                    i += 1;
                }
            }
            // 3) 复用缓冲并交换，避免重复分配/拷贝
            seq.resize(out_len);
            std::copy(out.begin(), out.begin() + static_cast<long>(out_len), seq.begin());
        }

        return seq;
    }

    py::tuple encode_with_limit_trace(const std::vector<int32_t> &seq_in, int32_t rank_limit) const {
        if (seq_in.empty()) return py::make_tuple(std::vector<int32_t>{}, std::vector<int32_t>{});
        std::vector<int32_t> seq = seq_in;
        std::vector<int32_t> out; out.resize(seq.size());
        std::vector<int32_t> ranks_applied;

        while (true) {
            int32_t best_rank = INT32_C(0x7fffffff);
            int32_t best_l = -1, best_r = -1;
            const size_t n = seq.size();
            if (n < 2) break;
            for (size_t i = 0; i + 1 < n; ++i) {
                int32_t l = seq[i];
                int32_t r = seq[i + 1];
                if (l >= 0 && static_cast<size_t>(l) < rank_tbl_.size()) {
                    const auto &mp = rank_tbl_[static_cast<size_t>(l)];
                    auto it = mp.find(r);
                    if (it != mp.end()) {
                        int32_t rank = it->second;
                        if (rank >= rank_limit) continue;
                        if (rank < best_rank) {
                            best_rank = rank;
                            best_l = l;
                            best_r = r;
                        }
                    }
                }
            }
            if (best_rank == INT32_C(0x7fffffff)) {
                break;
            }
            ranks_applied.push_back(best_rank);

            size_t out_len = 0;
            for (size_t i = 0; i < n; ) {
                if (i + 1 < n && seq[i] == best_l && seq[i + 1] == best_r) {
                    const auto &mpn = newid_tbl_[static_cast<size_t>(best_l)];
                    auto itn = mpn.find(best_r);
                    int32_t new_id = itn->second;
                    out[out_len++] = new_id;
                    i += 2;
                } else {
                    out[out_len++] = seq[i];
                    i += 1;
                }
            }
            seq.resize(out_len);
            std::copy(out.begin(), out.begin() + static_cast<long>(out_len), seq.begin());
        }
        return py::make_tuple(seq, ranks_applied);
    }

    std::vector<std::vector<int32_t>> batch_encode(const std::vector<std::vector<int32_t>> &seqs) const {
        std::vector<std::vector<int32_t>> results;
        results.reserve(seqs.size());
        for (const auto &s : seqs) {
            results.emplace_back(encode_with_limit(s, INT32_C(0x7fffffff)));
        }
        return results;
    }

    std::vector<std::vector<int32_t>> batch_encode_with_limit(const std::vector<std::vector<int32_t>> &seqs, int32_t rank_limit) const {
        std::vector<std::vector<int32_t>> results;
        results.reserve(seqs.size());
        for (const auto &s : seqs) {
            results.emplace_back(encode_with_limit(s, rank_limit));
        }
        return results;
    }

    // P2: ragged 批接口。输入为扁平 ids 与 offsets（长度 nseq+1，最后一个为 flat.size()）。
    // 返回 (flat_out, out_offsets)
    py::tuple encode_ragged(const std::vector<int32_t> &flat,
                            const std::vector<int32_t> &offsets) const {
        const size_t nseq = offsets.size() > 0 ? offsets.size() - 1 : 0;
        std::vector<int32_t> out_flat;
        out_flat.reserve(flat.size());
        std::vector<int32_t> out_offsets;
        out_offsets.reserve(offsets.size());
        out_offsets.push_back(0);

        for (size_t i = 0; i < nseq; ++i) {
            const size_t b = static_cast<size_t>(offsets[i]);
            const size_t e = static_cast<size_t>(offsets[i + 1]);
            std::vector<int32_t> seq;
            seq.assign(flat.begin() + static_cast<long>(b), flat.begin() + static_cast<long>(e));
            auto enc = encode(seq);
            out_flat.insert(out_flat.end(), enc.begin(), enc.end());
            out_offsets.push_back(static_cast<int32_t>(out_flat.size()));
        }
        return py::make_tuple(out_flat, out_offsets);
    }

private:
    // 左 token → (右 token → rank/newid)
    std::vector<std::unordered_map<int32_t, int32_t>> rank_tbl_;
    std::vector<std::unordered_map<int32_t, int32_t>> newid_tbl_;
};

PYBIND11_MODULE(_cpp_bpe, m) {
    py::class_<MinBPEEncoder>(m, "MinBPEEncoder")
        .def(py::init<const std::vector<int32_t>&,
                      const std::vector<int32_t>&,
                      const std::vector<int32_t>&,
                      const std::vector<int32_t>&>())
        .def("encode", &MinBPEEncoder::encode)
        .def("encode_with_limit", &MinBPEEncoder::encode_with_limit)
        .def("encode_with_limit_trace", &MinBPEEncoder::encode_with_limit_trace)
        .def("batch_encode", &MinBPEEncoder::batch_encode)
        .def("batch_encode_with_limit", &MinBPEEncoder::batch_encode_with_limit)
        .def("encode_ragged", &MinBPEEncoder::encode_ragged);

    // 训练：minBPE 逻辑（每轮全量统计 + 非重叠合并），返回 merge 规则与最终词表大小（不包含分隔符）。
    m.def("train_minbpe", [](const std::vector<std::vector<int32_t>> &seqs,
                               int32_t num_merges,
                               int32_t min_frequency) {
        // 1) 构建基础词表 & 拼接序列（使用分隔符防止跨序列合并）
        std::unordered_set<int32_t> base_vocab;
        base_vocab.reserve(1024);
        int32_t max_base_id = -1;
        for (const auto &s : seqs) {
            for (int32_t t : s) {
                base_vocab.insert(t);
                if (t > max_base_id) max_base_id = t;
            }
        }
        const int32_t separator = max_base_id + 1;
        int32_t next_id = separator + 1;

        std::vector<int32_t> ids;
        size_t total_len = 0;
        for (const auto &s : seqs) total_len += s.size();
        // 预估长度：序列间有分隔符（count = nseq-1）
        if (!seqs.empty()) total_len += (seqs.size() - 1);
        ids.reserve(total_len);
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (i > 0) ids.push_back(separator);
            const auto &s = seqs[i];
            ids.insert(ids.end(), s.begin(), s.end());
        }

        // 2) 主循环：每轮统计 pair 频次，选取最高频（同频词典序最小），然后非重叠合并
        std::vector<std::array<int32_t, 3>> merges;
        merges.reserve(static_cast<size_t>(num_merges));

        for (int32_t step = 0; step < num_merges; ++step) {
            // 统计相邻 pair（排除包含分隔符的）
            std::unordered_map<long long, int32_t> counts;
            counts.reserve(ids.size() / 2 + 1);
            const size_t n = ids.size();
            for (size_t i = 0; i + 1 < n; ++i) {
                int32_t l = ids[i];
                int32_t r = ids[i + 1];
                if (l == separator || r == separator) continue;
                long long key = (static_cast<long long>(static_cast<uint32_t>(l)) << 32)
                                | static_cast<uint32_t>(r);
                auto it = counts.find(key);
                if (it == counts.end()) counts.emplace(key, 1);
                else it->second += 1;
            }

            // 选择最佳 pair
            int32_t best_freq = -1;
            int32_t best_l = 0, best_r = 0;
            for (const auto &kv : counts) {
                int32_t freq = kv.second;
                if (freq < min_frequency) continue;
                long long key = kv.first;
                int32_t l = static_cast<int32_t>((key >> 32) & 0xFFFFFFFFLL);
                int32_t r = static_cast<int32_t>(key & 0xFFFFFFFFLL);
                if (freq > best_freq || (freq == best_freq && (l < best_l || (l == best_l && r < best_r)))) {
                    best_freq = freq;
                    best_l = l;
                    best_r = r;
                }
            }
            if (best_freq < min_frequency) {
                break;
            }

            // 应用一次非重叠合并： (best_l,best_r) -> new_id
            const int32_t new_id = next_id++;
            std::vector<int32_t> out;
            out.reserve(ids.size());
            for (size_t i = 0; i < ids.size(); ) {
                if (i + 1 < ids.size() && ids[i] == best_l && ids[i + 1] == best_r) {
                    out.push_back(new_id);
                    i += 2;
                } else {
                    out.push_back(ids[i]);
                    i += 1;
                }
            }
            ids.swap(out);
            merges.push_back({best_l, best_r, new_id});
        }

        const int32_t base_vocab_size = static_cast<int32_t>(base_vocab.size());
        const int32_t final_vocab_size = base_vocab_size + static_cast<int32_t>(merges.size());

        py::list py_merges;
        for (const auto &m3 : merges) py_merges.append(py::make_tuple(m3[0], m3[1], m3[2]));
        py::dict ret;
        ret[py::str("merge_rules")] = py_merges;
        ret[py::str("num_merges_performed")] = static_cast<int32_t>(merges.size());
        ret[py::str("final_vocab_size")] = final_vocab_size;
        ret[py::str("base_vocab_size")] = base_vocab_size;
        ret[py::str("separator_token")] = separator;
        return ret;
    }, py::arg("seqs"), py::arg("num_merges"), py::arg("min_frequency"));
}


