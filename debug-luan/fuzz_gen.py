def generate_fuzz_header(max_n):
    header_guard = "FUZZ_H"
    header_file = "fuzz.h"

    with open(header_file, 'w') as f:
        f.write(f"#ifndef {header_guard}\n")
        f.write(f"#define {header_guard}\n\n")

        f.write("#define ASSERT_XY_HELPER(x, y, N) \\\n")
        f.write("    do { \\\n")

        for n in range(5, max_n + 1):
            conditions = " || ".join([f"(y) != {i}" for i in range(1, n + 1)])
            f.write(f"        if (N == {n}) {{ assert((x) != 0 || {conditions}); }} \\\n")

        f.write("    } while (0)\n\n")
        f.write(f"#endif // {header_guard}\n")


# Example usage: generate a header file with N ranging from 5 to 9
generate_fuzz_header(30)
