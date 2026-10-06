#include <gtest/gtest.h>

#include <bgclib/BGCLib.hpp>

#include <algorithm>

namespace {

// Local model used only to prove that the public model contract accepts
// function-based extension through ADL. Real model tests should live next to
// their model-specific fixtures.
namespace contract_test {

struct Tag {};

struct Constants {
    PetscReal alpha {1.0};
};

using Parameters = Constants;

PetscErrorCode createState(Tag, const bgc::RunContext&, bgc::SimState&, const Constants&) {
    return PETSC_SUCCESS;
}

bgc::DiscreteOperator buildOperator(Tag, const bgc::RunContext&, const Constants&) {
    return {
        .layout = {
            .kind = bgc::MatrixLayoutKind::Scalar,
            .rows = 3,
            .cols = 3,
            .blockSize = 1,
        },
        .entries = {
            {.row = 0, .col = 0, .value = 1.0},
        },
    };
}

bgc::DiscreteRHS buildRHS(Tag, const bgc::RunContext&, const Constants&, PetscReal) {
    return {
        .size = 3,
        .entries = {
            {.row = 0, .value = 1.0},
        },
    };
}

} // namespace contract_test

} // namespace

namespace bgc {

template <>
struct ModelTraits<contract_test::Tag> {
    using Constants = contract_test::Constants;
    using Parameters = Constants;

    static constexpr std::string_view id {"CONTRACT_TEST"};
    static constexpr std::string_view name {"Contract Test Model"};
    static constexpr PetscInt fieldCount {1};
    static constexpr std::array<std::string_view, fieldCount> fieldNames {"u"};

    [[nodiscard]] static constexpr bool constantsAreValid(const Constants& constants) noexcept {
        return constants.alpha > 0.0;
    }

    [[nodiscard]] static constexpr bool parametersAreValid(const Parameters& parameters) noexcept {
        return constantsAreValid(parameters);
    }
};

} // namespace bgc

namespace {

TEST(Grid1DTest, ComputesUniformNodeCoordinates) {
    const bgc::Grid1D grid {
        .nx = 5,
        .length = 2.0,
        .x0 = -1.0,
    };

    EXPECT_TRUE(grid.isValid());
    EXPECT_DOUBLE_EQ(grid.dx(), 0.5);
    EXPECT_DOUBLE_EQ(grid.x(0), -1.0);
    EXPECT_DOUBLE_EQ(grid.x(4), 1.0);
}

TEST(TimeConfigTest, ComputesRoundedStepCount) {
    const bgc::TimeConfig time {
        .dt = 0.1,
        .finalTime = 1.0,
        .initialTime = 0.0,
    };

    EXPECT_TRUE(time.isValid());
    EXPECT_EQ(time.steps(), 10);
}

TEST(BoundaryConditionTest, StoresCanonicalLinearBoundaryData) {
    const auto dirichlet = bgc::BoundaryCondition::dirichlet(2.0);
    const auto neumann = bgc::BoundaryCondition::neumann([](const PetscReal t) {
        return 3.0 + t;
    });
    const auto robin = bgc::BoundaryCondition::robin(4.0, 5.0, 6.0);

    EXPECT_TRUE(dirichlet.isDirichlet());
    EXPECT_DOUBLE_EQ(dirichlet.alpha(), 1.0);
    EXPECT_DOUBLE_EQ(dirichlet.beta(), 0.0);
    EXPECT_DOUBLE_EQ(dirichlet.gamma(10.0), 2.0);

    EXPECT_TRUE(neumann.isNeumann());
    EXPECT_TRUE(neumann.isTimeDependent());
    EXPECT_DOUBLE_EQ(neumann.alpha(), 0.0);
    EXPECT_DOUBLE_EQ(neumann.beta(), 1.0);
    EXPECT_DOUBLE_EQ(neumann.gamma(0.5), 3.5);

    EXPECT_TRUE(robin.isRobin());
    EXPECT_DOUBLE_EQ(robin.alpha(), 4.0);
    EXPECT_DOUBLE_EQ(robin.beta(), 5.0);
    EXPECT_DOUBLE_EQ(robin.gamma(), 6.0);
}

TEST(ModelTraitsTest, DescribesBGCAtCompileTime) {
    using Tag = bgc::models::bgc::Tag;

    static_assert(bgc::ModelMetadata<Tag>);
    static_assert(bgc::modelFieldCount<Tag>() == 1);

    EXPECT_EQ(bgc::modelId<Tag>(), "BGC");
    EXPECT_EQ(bgc::ModelTraits<Tag>::fieldNames[0], "u");
    EXPECT_TRUE(bgc::ModelTraits<Tag>::constantsAreValid({.bv = 1.0}));
    EXPECT_TRUE(bgc::ModelTraits<Tag>::constantsAreValid({.bv = 0.0}));
    EXPECT_FALSE(bgc::ModelTraits<Tag>::constantsAreValid({.bv = -1.0}));
}

TEST(ModelTraitsTest, DescribesTBGCAtCompileTime) {
    using Tag = bgc::models::tbgc::Tag;

    static_assert(bgc::ModelMetadata<Tag>);
    static_assert(bgc::modelFieldCount<Tag>() == 2);

    EXPECT_EQ(bgc::modelId<Tag>(), "TBGC");
    EXPECT_EQ(bgc::ModelTraits<Tag>::fieldNames[0], "phi");
    EXPECT_EQ(bgc::ModelTraits<Tag>::fieldNames[1], "mu");
    EXPECT_TRUE(bgc::ModelTraits<Tag>::constantsAreValid({.bv = 1.0}));
    EXPECT_TRUE(bgc::ModelTraits<Tag>::constantsAreValid({.bv = 0.0}));
    EXPECT_FALSE(bgc::ModelTraits<Tag>::constantsAreValid({.bv = -1.0}));
}

TEST(ModelContractTest, AcceptsFreeFunctionBasedModels) {
    using Tag = contract_test::Tag;

    static_assert(bgc::ModelMetadata<Tag>);
    static_assert(bgc::ModelStateBuilder<Tag>);
    static_assert(bgc::ModelOperatorBuilder<Tag>);
    static_assert(bgc::ModelRHSBuilder<Tag>);
    static_assert(bgc::FullModel<Tag>);

    bgc::RunContext ctx;
    bgc::SimState state;
    const contract_test::Constants constants {.alpha = 2.0};

    EXPECT_EQ(createState(Tag {}, ctx, state, constants), PETSC_SUCCESS);
    EXPECT_TRUE(buildOperator(Tag {}, ctx, constants).isValid());
    EXPECT_TRUE(buildRHS(Tag {}, ctx, constants, 0.0).isValid());
}

TEST(ExactSolutionTest, EvaluatesFunctionWithModelConstants) {
    using Tag = bgc::models::bgc::Tag;

    const bgc::ExactSolution<Tag, PetscReal (*)(const bgc::ModelConstants<Tag>&,
                                                PetscReal,
                                                PetscReal)> exact {
        {.bv = 0.25},
        [](const bgc::ModelConstants<Tag>& constants, const PetscReal x, const PetscReal t) {
            return constants.bv * x * x + t;
        },
    };

    EXPECT_DOUBLE_EQ(exact(2.0, 0.5), 1.5);
    EXPECT_DOUBLE_EQ(exact.constants().bv, 0.25);
}

TEST(ManufacturedSourceTest, EvaluatesFunctionWithModelConstants) {
    using Tag = bgc::models::bgc::Tag;

    const bgc::ManufacturedSource<Tag, PetscReal (*)(const bgc::ModelConstants<Tag>&,
                                                     PetscReal,
                                                     PetscReal)> source {
        {.bv = 0.5},
        [](const bgc::ModelConstants<Tag>& constants, const PetscReal x, const PetscReal t) {
            return constants.bv * x + 2.0 * t;
        },
    };

    EXPECT_DOUBLE_EQ(source(3.0, 0.25), 2.0);
    EXPECT_DOUBLE_EQ(source.constants().bv, 0.5);
}

TEST(MMSCampaignTest, BuildsStandardOutputPathsAndObjective) {
    const bgc::MMSCampaign campaign {
        "MMS1BGC",
        "BGC",
        "Saida",
        {
            .setupCsv = "setup.csv",
            .convergenceCsv = "convergence.csv",
            .statusTxt = "status.txt",
            .summaryPrefix = "summary_N",
            .fieldsPrefix = "fields_N",
            .truncationPrefix = "lte_N",
        },
    };

    EXPECT_EQ(campaign.caseName(), "MMS1BGC");
    EXPECT_EQ(campaign.modelName(), "BGC");
    EXPECT_EQ(campaign.setupCsvPath(), std::filesystem::path("Saida") / "setup.csv");
    EXPECT_EQ(campaign.convergenceCsvPath(),
              std::filesystem::path("Saida") / "convergence.csv");
    EXPECT_EQ(campaign.statusPath(), std::filesystem::path("Saida") / "status.txt");
    EXPECT_NE(campaign.objective().find("truncation error"), std::string::npos);
}

TEST(BGCCoefficientsTest, ComputesInteriorStencilFromOldReferenceFormula) {
    const bgc::Grid1D grid {
        .nx = 8,
        .length = 1.0,
        .x0 = 0.0,
    };
    const bgc::models::bgc::Constants constants {.bv = 1.0e-2};

    const auto coeffs = bgc::models::bgc::computeCoefficients(grid, constants);

    EXPECT_EQ(coeffs.interior.ncols, 5);
    EXPECT_DOUBLE_EQ(coeffs.interior.coef[0], 5.4533333333333331);
    EXPECT_DOUBLE_EQ(coeffs.interior.coef[1], -29.813333333333333);
    EXPECT_DOUBLE_EQ(coeffs.interior.coef[2], 48.719999999999999);
    EXPECT_DOUBLE_EQ(coeffs.interior.coef[3], -29.813333333333333);
    EXPECT_DOUBLE_EQ(coeffs.interior.coef[4], 5.4533333333333331);

    const PetscReal sum = coeffs.interior.coef[0] +
                          coeffs.interior.coef[1] +
                          coeffs.interior.coef[2] +
                          coeffs.interior.coef[3] +
                          coeffs.interior.coef[4];
    EXPECT_NEAR(sum, 0.0, 1.0e-13);
}

TEST(BGCCoefficientsTest, MirrorsWestAndEastBoundaryClosures) {
    const bgc::Grid1D grid {
        .nx = 8,
        .length = 1.0,
        .x0 = 0.0,
    };
    const bgc::models::bgc::Constants constants {.bv = 1.0e-2};

    const auto coeffs = bgc::models::bgc::computeCoefficients(grid, constants);

    EXPECT_DOUBLE_EQ(coeffs.vol0.coef[0], coeffs.volNm1.coef[2]);
    EXPECT_DOUBLE_EQ(coeffs.vol0.coef[1], coeffs.volNm1.coef[1]);
    EXPECT_DOUBLE_EQ(coeffs.vol0.coef[2], coeffs.volNm1.coef[0]);

    EXPECT_DOUBLE_EQ(coeffs.vol1.coef[0], coeffs.volNm2.coef[3]);
    EXPECT_DOUBLE_EQ(coeffs.vol1.coef[1], coeffs.volNm2.coef[2]);
    EXPECT_DOUBLE_EQ(coeffs.vol1.coef[2], coeffs.volNm2.coef[1]);
    EXPECT_DOUBLE_EQ(coeffs.vol1.coef[3], coeffs.volNm2.coef[0]);
}

TEST(BGCBoundaryRHSTest, ComputesBoundarySourceFromOldReferenceFormula) {
    const bgc::Grid1D grid {
        .nx = 8,
        .length = 1.0,
        .x0 = 0.0,
    };
    const bgc::models::bgc::Constants constants {.bv = 1.0e-2};
    const bgc::BoundarySet boundaries {
        .west = {
            .conditions = {
                bgc::BoundaryCondition::dirichlet(2.0),
                bgc::BoundaryCondition::neumann(3.0),
            },
        },
        .east = {
            .conditions = {
                bgc::BoundaryCondition::dirichlet(5.0),
                bgc::BoundaryCondition::neumann(7.0),
            },
        },
    };

    const auto rhs = bgc::models::bgc::computeBoundaryRHS(grid, constants, boundaries, 0.0);
    const auto discrete = bgc::models::bgc::buildBoundaryRHS(grid, rhs);

    EXPECT_NEAR(rhs.vol0, 422.1923555555556, 1.0e-12);
    EXPECT_NEAR(rhs.vol1, 21.886044444444444, 1.0e-12);
    EXPECT_NEAR(rhs.volNm2, 38.900444444444446, 1.0e-12);
    EXPECT_NEAR(rhs.volNm1, 952.1635555555556, 1.0e-12);

    EXPECT_EQ(discrete.size, 8);
    EXPECT_EQ(discrete.entries.size(), 4U);
    EXPECT_EQ(discrete.entries[0].row, 0);
    EXPECT_EQ(discrete.entries[3].row, 7);
}

TEST(BGCOperatorTest, ExpandsStencilRowsIntoGenericOperatorEntries) {
    const bgc::Grid1D grid {
        .nx = 8,
        .length = 1.0,
        .x0 = 0.0,
    };
    const bgc::models::bgc::Constants constants {.bv = 1.0e-2};
    const auto coeffs = bgc::models::bgc::computeCoefficients(grid, constants);
    const auto op = bgc::models::bgc::buildOperator(grid, coeffs);

    EXPECT_TRUE(op.isValid());
    EXPECT_EQ(op.layout.rows, 8);
    EXPECT_EQ(op.layout.cols, 8);
    EXPECT_EQ(op.entries.size(), 34U);

    const auto row2 = std::ranges::count_if(op.entries, [](const bgc::OperatorEntry& entry) {
        return entry.row == 2;
    });
    EXPECT_EQ(row2, 5);

    const auto hasInteriorCenter = std::ranges::any_of(
        op.entries,
        [](const bgc::OperatorEntry& entry) {
            return entry.row == 2 &&
                   entry.col == 2 &&
                   entry.value == 48.719999999999999;
        });
    EXPECT_TRUE(hasInteriorCenter);
}

TEST(TBGCCoefficientsTest, ConvertsOldTBGCSReferenceFormulas) {
    const bgc::Grid1D grid {
        .nx = 8,
        .length = 1.0,
        .x0 = 0.0,
    };
    const bgc::TimeConfig time {
        .dt = 1.0e-3,
        .finalTime = 1.0e-3,
        .initialTime = 0.0,
    };
    const bgc::models::tbgc::Constants constants {.bv = 1.0e-2};

    const auto coeffs = bgc::models::tbgc::computeCoefficients(grid, time, constants);

    EXPECT_EQ(coeffs.A11.vol0.ncols, 3);
    EXPECT_DOUBLE_EQ(coeffs.A11.vol0.coef[0], 329.80000000000001);
    EXPECT_DOUBLE_EQ(coeffs.A11.vol0.coef[1], -15.170370370370371);
    EXPECT_DOUBLE_EQ(coeffs.A11.vol0.coef[2], 1.6384000000000001);
    EXPECT_DOUBLE_EQ(coeffs.A11.interior.coef[0], 125.0);

    EXPECT_EQ(coeffs.A12.interior.ncols, 3);
    EXPECT_DOUBLE_EQ(coeffs.A12.interior.coef[0], -8.0);
    EXPECT_DOUBLE_EQ(coeffs.A12.interior.coef[1], 16.0);
    EXPECT_DOUBLE_EQ(coeffs.A12.interior.coef[2], -8.0);

    EXPECT_EQ(coeffs.A21.vol0.ncols, 3);
    EXPECT_DOUBLE_EQ(coeffs.A21.vol0.coef[0], -2.92);
    EXPECT_DOUBLE_EQ(coeffs.A21.vol0.coef[1], 0.9955555555555555);
    EXPECT_DOUBLE_EQ(coeffs.A21.vol0.coef[2], -0.0768);
}

TEST(TBGCBoundaryRHSTest, ConvertsOldTBGCSBoundaryRHSFormulas) {
    const bgc::Grid1D grid {
        .nx = 8,
        .length = 1.0,
        .x0 = 0.0,
    };
    const bgc::models::tbgc::Constants constants {.bv = 1.0e-2};
    const bgc::BoundarySet boundaries {
        .west = {
            .conditions = {
                bgc::BoundaryCondition::dirichlet(2.0),
                bgc::BoundaryCondition::neumann(3.0),
            },
        },
        .east = {
            .conditions = {
                bgc::BoundaryCondition::dirichlet(5.0),
                bgc::BoundaryCondition::neumann(7.0),
            },
        },
    };

    const auto rhs = bgc::models::tbgc::computeBoundaryRHS(grid, constants, boundaries, 0.0);
    const auto discrete = bgc::models::tbgc::buildBoundaryRHS(grid, rhs);

    EXPECT_NEAR(rhs.b1.vol0, 456.6053925925925, 1.0e-12);
    EXPECT_NEAR(rhs.b2.vol0, -1.8744888888888889, 1.0e-12);
    EXPECT_NEAR(rhs.b1.volNm1, 989.7339259259257, 1.0e-12);
    EXPECT_NEAR(rhs.b2.volNm1, -5.304888888888889, 1.0e-12);

    EXPECT_EQ(discrete.size, 16);
    EXPECT_EQ(discrete.entries.size(), 4U);
    EXPECT_EQ(discrete.entries[0].row, 0);
    EXPECT_EQ(discrete.entries[1].row, 7);
    EXPECT_EQ(discrete.entries[2].row, 8);
    EXPECT_EQ(discrete.entries[3].row, 15);
}

TEST(TBGCOperatorTest, KeepsBlockStructureAndProvidesFlattenedOperator) {
    const bgc::Grid1D grid {
        .nx = 8,
        .length = 1.0,
        .x0 = 0.0,
    };
    const bgc::TimeConfig time {
        .dt = 1.0e-3,
        .finalTime = 1.0e-3,
        .initialTime = 0.0,
    };
    const bgc::models::tbgc::Constants constants {.bv = 1.0e-2};
    const auto coeffs = bgc::models::tbgc::computeCoefficients(grid, time, constants);
    const auto blocks = bgc::models::tbgc::buildBlockOperator(grid, coeffs);
    const auto op = bgc::models::tbgc::flattenBlockOperator(blocks);

    EXPECT_EQ(blocks.A11.layout.rows, 8);
    EXPECT_EQ(blocks.A12.layout.rows, 8);
    EXPECT_EQ(blocks.A21.layout.rows, 8);
    EXPECT_EQ(blocks.A22.layout.rows, 8);
    EXPECT_EQ(blocks.A11.entries.size(), 12U);
    EXPECT_EQ(blocks.A12.entries.size(), 22U);
    EXPECT_EQ(blocks.A21.entries.size(), 24U);
    EXPECT_EQ(blocks.A22.entries.size(), 8U);

    EXPECT_EQ(op.layout.kind, bgc::MatrixLayoutKind::Block);
    EXPECT_EQ(op.layout.rows, 16);
    EXPECT_EQ(op.layout.cols, 16);
    EXPECT_EQ(op.layout.blockSize, 2);
    EXPECT_EQ(op.entries.size(), 66U);

    const auto hasA12Row0Col8 = std::ranges::any_of(op.entries, [](const bgc::OperatorEntry& entry) {
        return entry.row == 0 && entry.col == 8 && entry.value == 32.0;
    });
    const auto hasA22Identity = std::ranges::any_of(op.entries, [](const bgc::OperatorEntry& entry) {
        return entry.row == 8 && entry.col == 8 && entry.value == 1.0;
    });
    EXPECT_TRUE(hasA12Row0Col8);
    EXPECT_TRUE(hasA22Identity);
}

} // namespace
