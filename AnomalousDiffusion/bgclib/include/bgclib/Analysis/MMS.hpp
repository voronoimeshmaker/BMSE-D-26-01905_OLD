#pragma once

// Generic Method of Manufactured Solutions campaign metadata.
//
// MMS cases across models share the same high-level objective:
//
//   analytical solution vs numerical solution vs truncation error
//
// This header keeps that objective in one reusable class. Model-specific
// programs provide the actual model constants, exact solution, source term,
// operator, RHS, and solver path.

#include <filesystem>
#include <string>
#include <string_view>

#include <petsc.h>

namespace bgc {

struct MMSOutputFiles {
    std::string setupCsv;
    std::string convergenceCsv;
    std::string statusTxt;
    std::string summaryPrefix;
    std::string fieldsPrefix;
    std::string truncationPrefix;
};

struct MMSErrorRecord {
    PetscInt  nx     {0};
    PetscReal h      {0.0};
    PetscReal dt     {0.0};
    PetscInt  steps  {0};
    PetscReal l1     {0.0};
    PetscReal l2     {0.0};
    PetscReal linf   {0.0};
    PetscReal lteL1  {0.0};
    PetscReal lteL2  {0.0};
    PetscReal lteInf {0.0};
};

class MMSCampaign {
public:
    MMSCampaign(std::string caseName,
                std::string modelName,
                std::filesystem::path outputDirectory,
                MMSOutputFiles outputFiles)
        : caseName_ {std::move(caseName)},
          modelName_ {std::move(modelName)},
          outputDirectory_ {std::move(outputDirectory)},
          outputFiles_ {std::move(outputFiles)} {}

    [[nodiscard]] std::string_view caseName() const noexcept {
        return caseName_;
    }

    [[nodiscard]] std::string_view modelName() const noexcept {
        return modelName_;
    }

    [[nodiscard]] const std::filesystem::path& outputDirectory() const noexcept {
        return outputDirectory_;
    }

    [[nodiscard]] const MMSOutputFiles& outputFiles() const noexcept {
        return outputFiles_;
    }

    [[nodiscard]] std::filesystem::path setupCsvPath() const {
        return outputDirectory_ / outputFiles_.setupCsv;
    }

    [[nodiscard]] std::filesystem::path convergenceCsvPath() const {
        return outputDirectory_ / outputFiles_.convergenceCsv;
    }

    [[nodiscard]] std::filesystem::path statusPath() const {
        return outputDirectory_ / outputFiles_.statusTxt;
    }

    [[nodiscard]] std::string objective() const {
        return "compare analytical solution, numerical solution, and truncation error";
    }

private:
    std::string           caseName_;
    std::string           modelName_;
    std::filesystem::path outputDirectory_;
    MMSOutputFiles        outputFiles_;
};

} // namespace bgc
