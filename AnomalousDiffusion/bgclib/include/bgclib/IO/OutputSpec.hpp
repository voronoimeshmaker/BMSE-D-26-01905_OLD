#pragma once

// -----------------------------------------------------------------------------
// OutputSpec.hpp
//
// Declarative output configuration for one simulation run.
//
// OutputSpec is intentionally only a data object. It describes where results
// should be written and which artifacts are enabled, while path construction and
// file ownership stay in the IO layer. This keeps programs and model classes from
// hard-coding filenames such as convergence.csv, fields.csv, or summary files.
//
// rootDir is the base directory configured by the user. runName is the logical
// case name below that base directory. The individual file fields may be left
// empty; IO helpers then choose conventional filenames for the enabled artifacts.
// -----------------------------------------------------------------------------

#include <string>

#include <petsc.h>

namespace bgc {

struct OutputSpec {
    // Base output directory, normally read from simulation.dat.
    std::string rootDir {"results"};

    // Optional case/subdirectory name, useful for MMS sweeps and parameter runs.
    std::string runName;

    // Optional explicit filenames for each output artifact.
    std::string fieldsFile;
    std::string errorsFile;
    std::string historyFile;
    std::string matrixFile;
    std::string summaryFile;

    // Per-artifact switches. Programs should honor these instead of deciding
    // locally whether a file is produced.
    bool writeFields  {true};
    bool writeErrors  {true};
    bool writeHistory {false};
    bool writeMatrix  {false};
    bool writeSummary {true};

    // Number of time steps between history writes for transient problems.
    PetscInt historyFrequency {1};
};

} // namespace bgc
