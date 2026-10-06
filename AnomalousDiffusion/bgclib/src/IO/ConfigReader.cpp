#include <bgclib/IO/ConfigReader.hpp>

#include <fstream>

namespace bgc {

std::string trim(const std::string_view text) {
    const auto first = std::find_if_not(text.begin(), text.end(), [](const unsigned char c) {
        return std::isspace(c) != 0;
    });
    if (first == text.end()) {
        return {};
    }

    const auto last = std::find_if_not(text.rbegin(), text.rend(), [](const unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    return {first, last};
}

std::string toLower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

std::vector<std::string> splitByComma(const std::string_view text) {
    std::vector<std::string> parts;
    std::stringstream stream {std::string {text}};
    std::string part;
    while (std::getline(stream, part, ',')) {
        std::string cleaned = trim(part);
        if (!cleaned.empty()) {
            parts.push_back(std::move(cleaned));
        }
    }
    return parts;
}

PetscReal parseReal(const std::string_view text) {
    const std::string cleaned = trim(text);
    const char* begin = cleaned.c_str();
    char* end = nullptr;
    const PetscReal value = static_cast<PetscReal>(std::strtod(begin, &end));
    if (end == begin) {
        throw std::runtime_error("Invalid PetscReal value: " + cleaned);
    }
    return value;
}

PetscInt parseInt(const std::string_view text) {
    const std::string cleaned = trim(text);
    const char* begin = cleaned.c_str();
    char* end = nullptr;
    const long value = std::strtol(begin, &end, 10);
    if (end == begin) {
        throw std::runtime_error("Invalid PetscInt value: " + cleaned);
    }
    return static_cast<PetscInt>(value);
}

bool parseBool(const std::string_view text) {
    const std::string cleaned = toLower(trim(text));
    if (cleaned == "true" || cleaned == "yes" || cleaned == "on" || cleaned == "1") {
        return true;
    }
    if (cleaned == "false" || cleaned == "no" || cleaned == "off" || cleaned == "0") {
        return false;
    }
    throw std::runtime_error("Invalid bool value: " + cleaned);
}

std::vector<PetscReal> parseRealList(const std::string_view text) {
    std::vector<PetscReal> values;
    for (const std::string& part : splitByComma(text)) {
        values.push_back(parseReal(part));
    }
    if (values.empty()) {
        throw std::runtime_error("Empty PetscReal list.");
    }
    return values;
}

std::vector<PetscInt> parseIntList(const std::string_view text) {
    std::vector<PetscInt> values;
    for (const std::string& part : splitByComma(text)) {
        values.push_back(parseInt(part));
    }
    if (values.empty()) {
        throw std::runtime_error("Empty PetscInt list.");
    }
    return values;
}

KeyValueConfig readKeyValueFile(const std::filesystem::path& path) {
    std::ifstream file {path};
    if (!file.is_open()) {
        throw std::runtime_error("Could not open configuration file: " + path.string());
    }

    KeyValueConfig data;
    std::string line;
    while (std::getline(file, line)) {
        if (const std::size_t comment = line.find('#'); comment != std::string::npos) {
            line.erase(comment);
        }

        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error("Invalid configuration line in " + path.string() + ": " +
                                     line);
        }

        std::string key = toLower(trim(std::string_view {line}.substr(0, eq)));
        std::string value = trim(std::string_view {line}.substr(eq + 1));
        if (!key.empty()) {
            data[std::move(key)] = std::move(value);
        }
    }
    return data;
}

bool containsKey(const KeyValueConfig& config, const std::string_view key) {
    return config.contains(toLower(std::string {key}));
}

std::string getStringOr(const KeyValueConfig& config,
                        const std::string_view key,
                        std::string fallback) {
    const auto it = config.find(toLower(std::string {key}));
    return it == config.end() ? std::move(fallback) : it->second;
}

PetscReal getRealOr(const KeyValueConfig& config,
                    const std::string_view key,
                    const PetscReal fallback) {
    const auto it = config.find(toLower(std::string {key}));
    return it == config.end() ? fallback : parseReal(it->second);
}

PetscInt getIntOr(const KeyValueConfig& config,
                  const std::string_view key,
                  const PetscInt fallback) {
    const auto it = config.find(toLower(std::string {key}));
    return it == config.end() ? fallback : parseInt(it->second);
}

bool getBoolOr(const KeyValueConfig& config, const std::string_view key, const bool fallback) {
    const auto it = config.find(toLower(std::string {key}));
    return it == config.end() ? fallback : parseBool(it->second);
}

} // namespace bgc
