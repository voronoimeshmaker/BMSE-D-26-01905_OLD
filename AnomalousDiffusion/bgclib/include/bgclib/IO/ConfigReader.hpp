#pragma once

// Common key/value configuration-file reader.
//
// The supported format is intentionally small and shared by the MMS programs:
//
//   key = value
//   # comments are ignored
//
// Keys are normalized to lower case. Values remain strings until a typed parser
// is requested. Lists are comma-separated and whitespace around items is
// ignored. This reader does not infer model semantics; model or driver code
// decides which keys are required and how defaults are applied.

#include <filesystem>
#include <unordered_map>
#include <vector>

#include <petsc.h>

namespace bgc {

using KeyValueConfig = std::unordered_map<std::string, std::string>;

[[nodiscard]] std::string trim(std::string_view text);
[[nodiscard]] std::string toLower(std::string text);
[[nodiscard]] std::vector<std::string> splitByComma(std::string_view text);

[[nodiscard]] PetscReal parseReal(std::string_view text);
[[nodiscard]] PetscInt parseInt(std::string_view text);
[[nodiscard]] bool parseBool(std::string_view text);
[[nodiscard]] std::vector<PetscReal> parseRealList(std::string_view text);
[[nodiscard]] std::vector<PetscInt> parseIntList(std::string_view text);

[[nodiscard]] KeyValueConfig readKeyValueFile(const std::filesystem::path& path);

[[nodiscard]] bool containsKey(const KeyValueConfig& config, std::string_view key);
[[nodiscard]] std::string getStringOr(const KeyValueConfig& config,
                                      std::string_view key,
                                      std::string fallback);
[[nodiscard]] PetscReal getRealOr(const KeyValueConfig& config,
                                  std::string_view key,
                                  PetscReal fallback);
[[nodiscard]] PetscInt getIntOr(const KeyValueConfig& config,
                                std::string_view key,
                                PetscInt fallback);
[[nodiscard]] bool getBoolOr(const KeyValueConfig& config,
                             std::string_view key,
                             bool fallback);

} // namespace bgc
