#ifndef DEBUG_LUAN_H
#define DEBUG_LUAN_H

#include <cstdlib>
#include <memory_resource>
// const bool DEBUG_LUAN = getenv("RAND_EST") ? true : false;

#define DEBUG_LUAN

#ifdef DEBUG_LUAN
#define FUZZ_LUAN
#define REORDER_RMW
// #define HALT_ON_ERROR
// #define POWER_SCHED
#endif

template<typename... Ts>
void Print(Ts&&... args) {
#ifdef DEBUG_LUAN
    static const int p = []() {
        auto ptr = std::getenv("Print");
        if (!ptr) return 0;
        return std::atoi(ptr);
        }();
    if (p != 1) return;

    [[maybe_unused]]
    auto ls = { ((llvm::outs() << args << ' '), 0)... };
    llvm::outs() << '\n';
    llvm::outs().flush();
#endif
}


#ifdef DEBUG_LUAN
inline std::pmr::unsynchronized_pool_resource mem_pool;
inline std::pmr::monotonic_buffer_resource mem_resource{ 65535 * 200, &mem_pool }; // size(Execution) ~ 104
#endif


#ifdef FUZZ_LUAN
// backward revisit
// #define FUZZ_BACKWARD
#endif

#define CATCHSIG_LUAN

#ifdef CATCHSIG_LUAN
// ...
#endif

// #define ENABLE_COLOR

#define RESET_COLOR         "\x1B[m"    
#define BACKGROUND_COLOR    "48;2;"
#define FOREGROUND_COLOR    "38;2;"

#define RGB(r, g, b, x) "\x1B[38;2;" #r ";" #g ";" #b "m" x RESET_COLOR


#ifdef ENABLE_COLOR
// #define RED(x) "\x1B[38;2;240;10;10m" x "\x1B[m"
#define RED(x)          RGB(240, 10, 10, x)
#define GREEN(x)        RGB(0, 250, 10, x) 
#define BLUE(x)         RGB(0, 190, 255, x)
#define YELLOW(x)       RGB(240, 250, 10, x) 
#else

#define RED(x)          x
#define GREEN(x)        x
#define BLUE(x)         x
#define YELLOW(x)       x
#endif  // ENABLE_COLOR

#endif