# C/C++ Knowledge Base

## Security Patterns

| Pattern | Risk | Fix |
|---|---|---|
| `sprintf(buf, fmt, user_input)` | Format String Attack (CRITICAL) | Use `snprintf` with fixed format |
| `strcpy(dst, src)` without length check | Buffer Overflow (CRITICAL) | Use `strncpy` or `std::string` |
| `gets(buf)` | Buffer Overflow (CRITICAL) | Use `fgets(buf, size, stdin)` |
| `system(user_input)` | Command Injection (CRITICAL) | Use `execvp` with argument array |
| `malloc` without null check | Null Dereference (HIGH) | Always check return value |
| `free(ptr); use(ptr)` | Use-After-Free (CRITICAL) | Set `ptr = NULL` after free |
| Raw `new`/`delete` | Memory Leak (HIGH) | Use `std::unique_ptr` / `std::shared_ptr` |
| `reinterpret_cast` on user data | Type Confusion (HIGH) | Use `static_cast` with validation |
| Hardcoded secrets | Secrets in Code (CRITICAL) | Use environment variables or config file |
| `rand()` for security | Weak RNG (HIGH) | Use `<random>` with `std::random_device` |

## Memory Safety

### RAII (Resource Acquisition Is Initialization)
```cpp
// BAD: manual resource management
FILE* f = fopen("data.txt", "r");
// ... if exception thrown here, f leaks
fclose(f);

// GOOD: RAII wrapper
auto f = std::unique_ptr<FILE, decltype(&fclose)>(fopen("data.txt", "r"), fclose);
// auto-closed on scope exit, even on exception
```

### Rule of Five
If a class manages resources, define ALL five:
1. **Destructor**: `~MyClass()`
2. **Copy constructor**: `MyClass(const MyClass&)`
3. **Copy assignment**: `MyClass& operator=(const MyClass&)`
4. **Move constructor**: `MyClass(MyClass&&) noexcept`
5. **Move assignment**: `MyClass& operator=(MyClass&&) noexcept`

Or use Rule of Zero: let `std::unique_ptr`/`std::shared_ptr`/`std::string`/`std::vector` manage resources.

### Smart Pointer Guide
| Use Case | Smart Pointer |
|---|---|
| Sole ownership | `std::unique_ptr<T>` |
| Shared ownership | `std::shared_ptr<T>` |
| Non-owning observer | `std::weak_ptr<T>` or raw pointer `T*` |
| Factory function | Return `std::unique_ptr<T>` |
| Container of polymorphic objects | `std::vector<std::unique_ptr<Base>>` |

## Modern C++ Idioms

| Anti-Pattern | Modern C++ |
|---|---|
| `#define CONSTANT 42` | `constexpr int CONSTANT = 42;` |
| `typedef struct { } Name;` | `struct Name { };` |
| `NULL` | `nullptr` |
| Raw arrays `int arr[10]` | `std::array<int, 10>` or `std::vector<int>` |
| `new T()` / `delete p` | `std::make_unique<T>()` |
| `void* userdata` callback | `std::function<>` or template |
| C-style cast `(int)x` | `static_cast<int>(x)` |
| `for (int i = 0; i < v.size(); i++)` | `for (const auto& item : v)` |
| Output parameter `void f(int* out)` | Return value: `int f()` or `std::optional<int> f()` |
| Header-only with `#ifndef` guards | `#pragma once` (widely supported) |

## Concurrency

| Anti-Pattern | Problem | Fix |
|---|---|---|
| `std::thread` without join/detach | Calls `std::terminate` | Always `.join()` or `.detach()`, or use `std::jthread` (C++20) |
| Shared data without mutex | Data race (undefined behavior) | `std::mutex` + `std::lock_guard` |
| Nested locks in inconsistent order | Deadlock | Use `std::scoped_lock(m1, m2)` for multiple locks |
| `volatile` for synchronization | Not thread-safe | Use `std::atomic<T>` |
| Detached thread accessing local vars | Use-after-free | Use `std::jthread` or ensure lifetime |
| Busy-wait loop | CPU waste | Use `std::condition_variable` |

## CMake Patterns

```cmake
# Minimum modern CMake
cmake_minimum_required(VERSION 3.14)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For clang-tidy

add_executable(myapp src/main.cpp src/util.cpp)
target_include_directories(myapp PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(myapp PRIVATE fmt::fmt)  # Modern: per-target linking

# Testing
enable_testing()
add_subdirectory(tests)
```

### Common CMake Errors
| Error | Fix |
|---|---|
| `Could not find package X` | Install dev package, or set `CMAKE_PREFIX_PATH` |
| `undefined reference to ...` | Add source file to `add_executable` / `add_library`, or link library |
| `No rule to make target` | File moved/renamed, update CMakeLists.txt |
| `target_link_libraries called with unknown target` | Define target before linking |

## Diagnostic Tools

```bash
# Static analysis
clang-tidy src/*.cpp -- -std=c++17
cppcheck --enable=all --std=c++17 src/

# Memory errors
valgrind --tool=memcheck --leak-check=full ./myapp
# or AddressSanitizer:
g++ -fsanitize=address -g -o myapp main.cpp && ./myapp

# Thread safety
g++ -fsanitize=thread -g -o myapp main.cpp && ./myapp

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build
ctest --test-dir build --output-on-failure
```

## Common Build Errors

| Error | Quick Fix |
|---|---|
| `undefined reference to 'X'` | Link the library: `target_link_libraries(... X)`, or add source file |
| `no matching function for call to 'X'` | Check argument types, add explicit conversion |
| `'X' was not declared in this scope` | Add `#include`, forward declaration, or check namespace |
| `multiple definition of 'X'` | Move definition to .cpp file, or add `inline` |
| `implicit conversion loses precision` | Add explicit `static_cast<>` |
| `template argument deduction failed` | Specify template arguments explicitly |
| `virtual function has incomplete return type` | Forward declare → full include |
| `expected ';' after class definition` | Add `;` after closing `}` of class/struct |
