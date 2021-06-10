#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// color printing code via:
// https://gist.github.com/zyvitski/fb12f2ce6bc9d3b141f3bd4410a6f7cf
enum class ansi_color_code : int {
  black = 30,
  red = 31,
  green = 32,
  yellow = 33,
  blue = 34,
  magenta = 35,
  cyan = 36,
  white = 37,
  bright_black = 90,
  bright_red = 91,
  bright_green = 92,
  bright_yellow = 93,
  bright_blue = 94,
  bright_magenta = 95,
  bright_cyan = 96,
  bright_white = 97,
};
template <typename printable>
std::string print_as_color(printable const &value, ansi_color_code color) {
  std::stringstream sstr;
  sstr << "\033[1;" << static_cast<int>(color) << "m" << value << "\033[0m";
  return sstr.str();
}
struct Representable {
  virtual void printHeader() = 0;
  virtual void printData(void *ptr) = 0;
};

struct SpaceHandle {
  char name[64];
};

struct SH {
  char name[64];
};

struct region_stack {
  std::vector<std::string> regions;
  std::string represent() {
    std::string built = "";
    int depth = 0;
    for (auto region : regions) {
      for (int x = 0; x < depth; ++x) {
        built += " ";
      }
      built += region + "\n";
      ++depth;
    }
    return built;
  }
};

std::vector<Representable *> histories;
using map_type = std::map<void *, region_stack>;
map_type ptr_map;

struct allocHistory : public Representable {
  void printHeader() override {
    std::cout << print_as_color("==============\n", ansi_color_code::blue);
    std::cout << print_as_color("Allocated at\n", ansi_color_code::blue);
    std::cout << print_as_color("==============\n", ansi_color_code::blue);
  }
  void printData(void *in_ptr) override {
    void *ptr = in_ptr - 128;
    std::cout << print_as_color(ptr_map[ptr].represent(),
                                ansi_color_code::yellow);
  }
};

region_stack current_regions;
struct DeepCopyHistory {
  struct Entry {
    std::string other_end;
    region_stack code_location;
    std::string space;
    void print() {
      std::cout << print_as_color("View: ", ansi_color_code::blue) << other_end
                << std::endl;
      std::cout << print_as_color("Space: ", ansi_color_code::blue) << space
                << std::endl;
      std::cout << print_as_color("Region stack (next line)\n",
                                  ansi_color_code::blue);
      std::cout << print_as_color(code_location.represent(),
                                  ansi_color_code::yellow);
      std::cout << std::flush;
    }
  };
  void print() {
    for (auto entry : history) {
      entry.print();
    }
  }
  std::vector<Entry> history;
};
struct DeepCopyTracker : public Representable {
  std::map<const void *, DeepCopyHistory> src_history;
  std::map<const void *, DeepCopyHistory> dst_history;
  void printHeader() override {
    std::cout << print_as_color("==============\n", ansi_color_code::blue);
    ;
    std::cout << print_as_color("Deep Copies\n", ansi_color_code::blue);
    ;
    std::cout << print_as_color("==============\n", ansi_color_code::blue);
    ;
  }
  void printData(void *in_ptr) override {
    auto ptr = in_ptr;
    std::cout << print_as_color("Copies from this view: \n",
                                ansi_color_code::green)
              << std::endl;
    src_history[ptr].print();
    std::cout << print_as_color("Copies to this view: \n",
                                ansi_color_code::green)
              << std::endl;
    dst_history[ptr].print();
  }
  void register_deep_copy(SpaceHandle dst_handle, const char *dst_name,
                          const void *dst_ptr, SpaceHandle src_handle,
                          const char *src_name, const void *src_ptr,
                          uint64_t size) {
    src_history[dst_ptr].history.push_back(DeepCopyHistory::Entry{
        std::string(src_name), current_regions, std::string(dst_handle.name)});
    dst_history[src_ptr].history.push_back(DeepCopyHistory::Entry{
        std::string(dst_name), current_regions, std::string(src_handle.name)});
  }
};
DeepCopyTracker *deep_copies;
void printData(void *ptr) {

  for (auto *history : histories) {
    history->printHeader();
    history->printData(ptr);
  }
}
extern "C" void kokkosp_init_library(int loadseq, uint64_t version,
                                     uint32_t ndevinfos, void *devinfos) {
  histories.push_back(new allocHistory());
  deep_copies = new DeepCopyTracker();
  histories.push_back(deep_copies);
}
extern "C" void kokkosp_allocate_data(SH h, const char *name, void *ptr,
                                      uint64_t size) {
  static int count;
  ptr_map[ptr] = current_regions;
}
extern "C" void kokkosp_deallocate_data(SH h, const char *name, void *ptr,
                                        uint64_t size) {
  static int count;
}
extern "C" void kokkosp_push_profile_region(const char *region) {
  current_regions.regions.push_back(std::string(region));
}
extern "C" void kokkosp_pop_profile_region() {
  current_regions.regions.pop_back();
}
extern "C" void kokkosp_begin_deep_copy(SpaceHandle dst_handle,
                                        const char *dst_name,
                                        const void *dst_ptr,
                                        SpaceHandle src_handle,
                                        const char *src_name,
                                        const void *src_ptr, uint64_t size) {
  deep_copies->register_deep_copy(dst_handle, dst_name, dst_ptr, src_handle,
                                  src_name, src_ptr, size);
}
