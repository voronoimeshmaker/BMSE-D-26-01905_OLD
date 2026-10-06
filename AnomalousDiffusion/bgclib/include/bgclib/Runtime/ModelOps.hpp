#pragma once

// Runtime function-table interface for models.
//
// bgclib's primary model API is compile-time: each model has a tag, traits,
// constants, and free functions. Runtime programs sometimes need to choose a
// model from an input string, though. This header is reserved for that boundary:
// a table of model operations can map a runtime ModelId to the concrete
// compile-time implementation without introducing a virtual base class.
//
// Current status: the runtime selection layer is intentionally thin while MMS
// programs use direct model calls. Add concrete operation-table types here when
// a driver genuinely needs model selection from configuration files.
