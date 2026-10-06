#pragma once

// Shared routines for writing fields, matrices, histories, errors, and summaries.

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace bgc {

[[nodiscard]] std::ofstream openOutputFile(const std::filesystem::path& path);

template <typename WriteRows>
void writeTextTable(const std::filesystem::path& path,
                    const std::string_view header,
                    WriteRows&& writeRows,
                    const int precision = 16) {
    std::ofstream file = openOutputFile(path);
    file << std::scientific << std::setprecision(precision);
    if (!header.empty()) {
        file << header << '\n';
    }
    std::forward<WriteRows>(writeRows)(file);
}

} // namespace bgc
