#include <bgclib/IO/ResultWriter.hpp>
#include <bgclib/IO/OutputPaths.hpp>


namespace bgc {

// Opens text output with automatic flushing. MMS and long physical campaigns
// can then be monitored while they run, even before files are closed.
std::ofstream openOutputFile(const std::filesystem::path& path) {
    ensureParentDirectory(path);
    std::ofstream file {path};
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + path.string());
    }
    file << std::unitbuf;
    return file;
}

} // namespace bgc
