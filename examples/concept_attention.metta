; MeTTa sketch of the same cognitive cycle.
; Requires: pip install hyperon-ecan[hyperon]
; Then:     metta-py examples/concept_attention.metta
;
; The Python grounded atoms live in hyperon_ecan.ext.

!(import! &self hyperon_ecan)

!(ecan-new)
!(ecan-fact Inheritance dog mammal)
!(ecan-fact Inheritance mammal animal)
!(ecan-fact Inheritance cat mammal)
!(ecan-fact Inheritance oak tree)
!(ecan-add wolf)

!(ecan-tick dog)
!(ecan-tick dog)
!(ecan-focus)
!(neural-similar dog)
!(ecan-sti wolf)
!(ecan-sti oak)
