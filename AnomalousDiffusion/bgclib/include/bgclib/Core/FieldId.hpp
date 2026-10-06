#pragma once

// Stable field identifiers used to access model fields from drivers.
//
// This header is reserved for a small typed identifier layer that will allow
// runtime code to ask for fields by semantic name instead of by raw integer
// offsets. The current MMS programs mostly use explicit flattened vectors:
//
//   scalar: [ phi ]
//   TBGC:   [ phi ... mu ]
//   TSOM:   [ U ... V ]
//
// When the runtime layer grows, field ids should describe those names without
// changing the model-specific unknown ordering documented by each model.
