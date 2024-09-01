#ifndef GRAPH_CCOUNTER_H
#define GRAPH_CCOUNTER_H


#include <filesystem>
#include <fstream>
#include <string_view>
#include <list>
#include <random>
#include <string>
#include "ExecutionGraph.hpp"
#include "EventLabel.hpp"

namespace {
    template <class T>
    inline void hash_combine(std::size_t& seed, const T& v) {

        seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
}

template<>
struct std::hash<EventLabel*> {
    std::size_t operator()(const EventLabel* lab) const {
        BUG_ON(!lab);
        std::size_t h = 0;
        // kind, ordering, 
        auto pos = lab->getPos();
        hash_combine(h, pos.thread);
        hash_combine(h, pos.index);
        // auto kind = lab->getOrdering();
        // hash_combine(h, static_cast<std::underlying_type_t<decltype(kind)>>(kind));
        // write value
        if (auto w = dynamic_cast<const WriteLabel*>(lab)) {
            // hash_combine(h, w->getVal().get());
        }
        // rf
        if (auto r = dynamic_cast<const ReadLabel*>(lab)) {
            if (auto rf = r->getRf())
                hash_combine(h, (*this)(rf));
        }
        return h;
    }
};


struct GraphHash {
    std::size_t operator()(const ExecutionGraph& g) const {
        // Print("hashing g...");
        std::size_t h = 0;
        for (auto i = 0u; i < g.getNumThreads(); i++) {
            // for each thread...
            // Print("hashing thread", i);
            for (auto j = 0u; j < g.getThreadSize(i); j++) {
                // for each event in thread i
                const EventLabel* lab = g.getEventLabel(Event(i, j));
                BUG_ON(!lab);
                hash_combine(h, std::hash<EventLabel*>{}(lab));
            }
        }
        // Print("hash of g:", h);
        return h;
    }
};




struct GraphCounter {

    size_t N;
    std::vector<size_t> plot{};
    inline static GraphHash ghash{};
    // std::unordered_map<size_t, size_t> graph_freq{};

    std::unordered_map<size_t, size_t> graph_complete_freq{};

    std::unordered_map<size_t, size_t> graph_block_freq{};

    std::string get_postfix() const {
        using namespace std::string_literals;
        auto s = std::getenv("MUTATION");
        BUG_ON(!s);
        // Print("MUTATION =", s);
        auto mut = std::atoi(s);
        // BUG_ON(mut < 0 || mut > 4);
        // Print("mut =", mut);
        if (mut == 0) return "-rnd";
        if (mut == 5) return "-mix";
        // for others:
        return "-fz"s + std::to_string(mut);
    }

    // #define SAVE_GRAPHS

    std::unordered_map<size_t, std::unique_ptr<ExecutionGraph>> graphs;
    auto log(const ExecutionGraph& g, bool isBlock) {
        // Print("logging graph\n");
        static int i = 0;
        auto h = ghash(g);
        float freq{};
        if (isBlock) {
            freq =
                ++graph_block_freq[h];
        }
        else {
            freq =
                ++graph_complete_freq[h];
        }
        sum_freq += 2 * freq - 1;
        // print_gmap();
        // BUG_ON(i >= N);
        plot.push_back(
            graph_complete_freq.size()
            + graph_block_freq.size()
        );
        ++i;
#ifdef SAVE_GRAPHS
        if (!graphs.contains(h)) {
            graphs[h] = g.clone();
        }
#endif  
        constexpr int part = 10;
        if (auto m = plot.size() % (N / part); m == 0) {
            const auto num_g = plot.back();
            const auto num_it = plot.size();
            auto percent = 100.0 * num_g / num_it;
            llvm::outs() << plot.size() / (N / part) << " of " << part << " done " << num_g << "/" << num_it << "(=" << llvm::format("%.2f", percent) << "%)" << '\n';
        }
        return h;
    }



    float sum_freq = 0.0;  // increment = x^2 - (x-1)^2 = 2x-1
    double relative_freq(size_t h) {
        auto freq = graph_block_freq.contains(h) ? graph_block_freq[h] : graph_complete_freq[h];
        BUG_ON(freq == 0);
        float w = freq * freq;

        int cnt = graph_complete_freq.size()
            + graph_block_freq.size();

        if (cnt == 0) return 0;
        auto res = w / (sum_freq / cnt);
        return res;
    }

    bool contains(const ExecutionGraph& g) const {
        // Print("check contains g or not\n");
        auto h = ghash(g);
        return graph_complete_freq.contains(h) || graph_block_freq.contains(h);
    }
    std::string test_name;
    std::string out_dir = "out";
    // GraphCounter() {}
    GraphCounter(size_t N, const std::string& test) :
        test_name(std::filesystem::path(test).filename().stem()),
        N(N)
    {

#ifdef FUZZ_LUAN
        test_name += get_postfix();
#else
        test_name += "-rnd";
#endif
        // Print("constructing counter, N = ", N);
        plot.reserve(N + 2);

    }
    GraphCounter(size_t N, const std::string& test, const std::string& out) :
        test_name(std::filesystem::path(test).filename().stem()),
        out_dir(out),
        N(N)
    {


#ifdef FUZZ_LUAN
        test_name += "-fz";
#ifdef FUZZ_BACKWARD
        test_name += "_b";
#endif  // FUZZ_BACKWARD
#else
        test_name += "-rnd";
#endif
        // Print("constructing counter, N = ", N);
        plot.reserve(N);


    }


    // make it a global variable
    void init(size_t N, const std::string& test) {
        static bool initialized = false;
        if (!initialized) {
            N = N;
            test_name = test;
            test_name += get_postfix();
            plot.reserve(N + 2);
            initialized = true;
        }
    }

    ~GraphCounter() {
        // print_gmap();
        // print_coverage();
        if (!getenv("NO_SAVE_COVERAGE"))
            save_coverage_plot();
        llvm::outs() << "complete: " << graph_complete_freq.size() << ", block: " << graph_block_freq.size() << '\n';
        llvm::outs() << "uniq(of N): " << plot[N - 1] << '\n';
        llvm::outs() << "uniq(seen): " << plot.back() << "/" << plot.size() << '\n';
#ifdef SAVE_GRAPHS
        for (auto& [h, p] : graphs) {
            if (graph_block_freq.contains(h)) {
                Print("---", h, "(blocked)---");
                Print(*p);
            }
        }
#endif
        print_bug();
    }

    std::vector<int> bug_iters{};

    void bug(int i) {
        bug_iters.push_back(i);
    }
    void bug() { bug(plot.back() + 1); }

    void print_bug() {
        llvm::outs() << "first bug iter: " << (bug_iters.size() ? bug_iters[0] : -1) << '\n';
    }

    void print_gmap() const {
        Print("graph map size = ", graph_complete_freq.size() + graph_block_freq.size());
        for (auto&& g : graph_complete_freq) {
            Print(g.first, ": ", g.second);
        }
        for (auto&& g : graph_block_freq) {
            Print(g.first, ": ", g.second);
        }
    }
    void print_coverage() const {
        for (int i = 0; i < plot.size(); i++) {
            Print(i, ": ", plot[i]);
        }
    }

    void save_coverage_plot() const {
        using namespace std::filesystem;
        // Print("test: ", test_name);
        // Print("curr: ", current_path());
        auto p = current_path() / path(out_dir);
        if (!exists(p)) {
            create_directories(p);
        }
        p /= (test_name + ".txt");

        // Print("p = ", p.c_str());
        llvm::outs() << "saving to " << p.c_str() << '\n';
        std::ofstream f(p);
        for (int i = 0; i < plot.size(); i++) {
            f << i << " " << plot[i] << '\n';
            // getchar();
        }

    }



    using PrefixT = std::unique_ptr<ExecutionGraph>;
    template<typename T> using QueueT = std::list<T>;
    // std::unordered_map<PrefixT, float> prefixes;
    // TODO: make it a tuple
    QueueT<PrefixT> prefixes;
    QueueT<float> weights;
    QueueT<size_t> prefixHashes;
    const float INIT_SCORE = 5000.0; // prioritize unchosen seeds

    bool isSampling = true;
    int sampleCnt = 0;      // TODO: set threshold

    bool is_sampling() const {
        return  isSampling;
    }



    const int PBUF_SIZE = []() {
        auto p = std::getenv("PBUF_SIZE");
        return p ? std::atoi(p) : 400;
        }();


    const int SAMPLE_TIME = []() {
        auto p = std::getenv("SAMPLE_TIME");
        return p ? std::atoi(p) : 0;
        }();

    const int REWARDF = []() {
        auto p = std::getenv("REWARDF");
        return p ? std::atoi(p) : 3;
        }();

    void addPrefix(std::unique_ptr<ExecutionGraph> p) {
        if (isSampling) {
            sampleCnt++;
            if (sampleCnt >= SAMPLE_TIME) isSampling = false;
        }
        // TODO: constrain queue size (drop out)
        auto h = ghash(*p);
        auto it = std::find_if(begin(prefixes), end(prefixes), [h](const PrefixT& p_) {return h == ghash(*p_); });
        if (it == end(prefixes)) {
            prefixes.push_back(std::move(p));
            weights.push_back(INIT_SCORE);
            prefixHashes.push_back(h);
        }
        if (prefixes.size() > PBUF_SIZE) {
            auto minWeightIt = std::min_element(weights.begin(), weights.end());
            // delete min weight
            auto prefixIt = prefixes.begin();
            auto weightIt = weights.begin();
            auto hashIt = prefixHashes.begin();
            while (weightIt != minWeightIt) {
                ++prefixIt;
                ++weightIt;
                ++hashIt;
            }
            prefixes.erase(prefixIt);
            weights.erase(weightIt);
            prefixHashes.erase(hashIt);
        }
    }

    static inline std::mt19937 prefixRng{ 42 };

    PrefixT pickPrefix() {
        if (prefixes.size() == 0) { return nullptr; }
        // randomly pick a seed based on their weights
        std::discrete_distribution<> dist{ weights.begin(), weights.end() };
        auto idx = dist(prefixRng);
        // auto idx = 1;
        Print("got index:", idx);
        auto pit = std::next(prefixes.begin(), idx);
        auto wit = weights.begin(); advance(wit, idx);
        if (*wit == INIT_SCORE) {
            *wit = 0.0;
        }
        return (*pit)->clone();
    }

    std::optional<int> currPrefixIdx;
    void locatePrefix(const ExecutionGraph& g) {
        // check which prefix g is
        Print("before rerun, g =", g);
        // getchar();
        auto h = ghash(g);
        auto it = prefixHashes.begin();
        auto pit = prefixes.begin();
        for (int i = 0; i < prefixHashes.size(); i++) {
            if (h == *it) {
                currPrefixIdx = i;
                Print("found prefix:", *(*pit));
                // getchar();
                return;
            }
            ++it;
            ++pit;
        }
        Print("didn't found corresponding prefix");
        // getchar();
    }



    void updateWeights(bool isNew, size_t hash) {
        // naive: new graph -> weight++ , old graph -> weight--
        // call this in checking gcounter.contains(g) {...}
        if (currPrefixIdx) {
            auto idx = *currPrefixIdx;
            auto it = std::next(weights.begin(), idx);
            auto pit = std::next(prefixes.begin(), idx);
            Print("updating weights for:", *(*pit));
            // getchar();

            auto x = *it;
            if (REWARDF == 0) {
                if (isNew) *it += 2;
                else *it -= 2;
            }
            if (REWARDF == 1) {
                if (isNew) {
                    // *it += std::log(1 + std::exp(x));
                    *it += std::exp(x);
                }
                else {
                    *it -= std::exp(-x);
                }
            }
            if (REWARDF == 2) {
                if (isNew) {
                    *it += std::log(1 + std::exp(x));
                }
                else {
                    *it -= std::log(1 + std::exp(-x));
                }
            }
            if (REWARDF == 3) {
                if (isNew) {
                    auto y = std::log(1 + std::exp(x));
                    *it += std::min<double>(y, 30.0);
                }
                else {
                    auto y = std::log(1 + std::exp(-x));
                    *it -= std::min<double>(y, 30.0);
                }
            }
            if (REWARDF == 4) {
                auto thres = INIT_SCORE;
                if (isNew) *it += thres;
                else {
                    auto freq = graph_block_freq.contains(hash) ? graph_block_freq[hash] : graph_complete_freq[hash];
                    auto w = freq * freq;
                    double sum = 0.0;
                    int cnt = 0;
                    for (auto&& b : graph_block_freq) {
                        sum += b.second * b.second;
                        cnt++;
                    }
                    for (auto&& b : graph_complete_freq) {
                        sum += b.second * b.second;
                        cnt++;
                    }
                    sum /= cnt;
                    *it -= thres * (w / sum);
                }
            }

        }

    }





};


struct EventCounter {
    size_t n_totalEvents[3];
    size_t n_remainedEvents[3];
    size_t n_graph; // graphs that are mutated

    std::string test_name;
    std::string out_dir = "out";

    EventCounter(const std::string& test) :
        test_name(std::filesystem::path(test).filename().stem())
    {}

    ~EventCounter() {
        using namespace std::filesystem;
        std::string out_dir = "out";
        auto p = current_path() / path(out_dir);
        if (!exists(p)) {
            create_directories(p);
        }
        p /= (test_name + "-evcnt.txt");

        // Print("p = ", p.c_str());
        llvm::outs() << "saving to " << p.c_str() << '\n';
        std::ofstream f(p);
        f << "num of graphs: " << n_graph << "\n";
        f << "cut \ttotal events \tremained events\n";
        const char* cuts[] = { "revisit", "minimal", "maximal" };
        for (int i = 0; i < 3; i++) {
            f << cuts[i] << '\t';
            f << n_totalEvents[i] << '\t';
            f << n_remainedEvents[i] << '\n';
        }


    }
};


#endif