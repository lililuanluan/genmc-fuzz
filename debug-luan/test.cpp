#include <iostream>
#include <thread>


void t() {
    std::cout << "hello\n";
}

int main() {
    std::thread t1(t);
    std::thread t2(t);
    t1.join();
    t2.join();
}