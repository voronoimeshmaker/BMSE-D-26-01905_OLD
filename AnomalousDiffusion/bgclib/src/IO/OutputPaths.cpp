#include <bgclib/IO/OutputPaths.hpp>

namespace bgc {

std::filesystem::path caseDirectoryFromSource(const std::string_view sourceFile) {
    return std::filesystem::path {sourceFile}.parent_path();
}

std::filesystem::path resolveOutputDirectory(const std::filesystem::path& caseDir,
                                             const std::filesystem::path& configuredDir) {
    if (configuredDir.is_absolute()) {
        return configuredDir;
    }
    return caseDir / configuredDir;
}

void ensureDirectory(const std::filesystem::path& directory) {
    if (directory.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        throw std::runtime_error("Could not create directory: " + directory.string() + " (" +
                                 ec.message() + ")");
    }
}

void ensureParentDirectory(const std::filesystem::path& filePath) {
    ensureDirectory(filePath.parent_path());
}

std::string realDirectoryName(const std::string_view prefix, const PetscReal value) {
    std::ostringstream out;
    out << prefix << std::scientific << std::setprecision(3) << value;
    std::string text = out.str();
    for (char& c : text) {
        if (c == '+') {
            c = 'p';
        } else if (c == '-') {
            c = 'm';
        } else if (c == '.') {
            c = 'p';
        }
    }
    return text;
}

std::filesystem::path bvOutputDirectory(const std::filesystem::path& outputDir,
                                        const PetscReal bv) {
    return outputDir / realDirectoryName("Bv_", bv);
}

std::filesystem::path nxOutputDirectory(const std::filesystem::path& outputDir,
                                        const PetscInt nx) {
    return outputDir / ("N" + std::to_string(static_cast<int>(nx)));
}

std::filesystem::path bvNxOutputDirectory(const std::filesystem::path& outputDir,
                                          const PetscReal bv,
                                          const PetscInt nx) {
    return nxOutputDirectory(bvOutputDirectory(outputDir, bv), nx);
}

std::string nxFileName(const std::string_view prefix,
                       const PetscInt nx,
                       const std::string_view extension) {
    return std::string {prefix} + std::to_string(static_cast<int>(nx)) + std::string {extension};
}

} // namespace bgc
