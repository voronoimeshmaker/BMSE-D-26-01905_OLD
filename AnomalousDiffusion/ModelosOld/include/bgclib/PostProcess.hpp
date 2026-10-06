#pragma once

#include <petsc.h>

#include <bgclib/Analytics.hpp>
#include <bgclib/SimConfig.hpp>
#include <bgclib/SimState.hpp>

namespace bgc {

struct ErrorNorms {
    PetscReal L1   = 0.0;
    PetscReal L2   = 0.0;
    PetscReal Linf = 0.0;
};

PetscErrorCode computeErrorNorms(const SimConfig& cfg,
                                 SimState&        st,
                                 PetscReal        t,
                                 const FieldFn&   phiFn,
                                 ErrorNorms&      norms);

PetscErrorCode checkThermodynamicConsistency(const SimConfig& cfg,
                                             SimState&        st,
                                             PetscReal        t,
                                             PetscReal&       F_prev,
                                             PetscBool&       violated);

} // namespace bgc