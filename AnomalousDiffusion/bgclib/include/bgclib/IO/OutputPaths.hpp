#pragma once

// Resolves OutputSpec into concrete output directories and file paths.

#include <filesystem>

#include <petsc.h>

namespace bgc {

[[nodiscard]] std::filesystem::path caseDirectoryFromSource(std::string_view sourceFile);
[[nodiscard]] std::filesystem::path resolveOutputDirectory(
    const std::filesystem::path& caseDir,
    const std::filesystem::path& configuredDir);

void ensureDirectory(const std::filesystem::path& directory);
void ensureParentDirectory(const std::filesystem::path& filePath);

[[nodiscard]] std::filesystem::path bvOutputDirectory(const std::filesystem::path& outputDir,
                                                      PetscReal bv);
[[nodiscard]] std::filesystem::path nxOutputDirectory(const std::filesystem::path& outputDir,
                                                      PetscInt nx);
[[nodiscard]] std::filesystem::path bvNxOutputDirectory(const std::filesystem::path& outputDir,
                                                        PetscReal bv,
                                                        PetscInt nx);

[[nodiscard]] std::string nxFileName(std::string_view prefix,
                                     PetscInt nx,
                                     std::string_view extension);
[[nodiscard]] std::string realDirectoryName(std::string_view prefix, PetscReal value);

} // namespace bgc
