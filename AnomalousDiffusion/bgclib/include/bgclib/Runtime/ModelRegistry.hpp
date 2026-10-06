#pragma once

// Registry mapping ModelId values to runtime model operations.
//
// The registry is the planned lookup point for programs that read a model name
// such as "BGC", "TBGC", or "TSOM" from input files. It should translate that
// string into a non-owning operation table defined in ModelOps.hpp.
//
// The registry must not contain coefficient formulas. It should only connect
// names to already-documented model APIs.
