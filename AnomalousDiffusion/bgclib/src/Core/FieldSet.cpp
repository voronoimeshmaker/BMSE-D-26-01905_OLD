// FieldSet is a reserved core abstraction for grouped PETSc field storage.
//
// The MMS programs currently work directly with PETSc Vec objects and model
// ordering conventions. When production runs need named field views or block
// accessors, their non-template implementation should live here.
