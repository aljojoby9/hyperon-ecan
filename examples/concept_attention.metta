; Walk through ECAN from MeTTa, one step at a time.
; pip install hyperon-ecan[hyperon]
; metta examples/concept_attention.metta
;
; Each ! line prints a result. Comments say what that result means.

!(import! &self hyperon_ecan)
!(ecan-new)

; --- knowledge -----------------------------------------------------------
; Inheritance facts also add Hebbian links. wolf has no symbolic edge.
!(ecan-fact Inheritance dog mammal)
!(ecan-fact Inheritance mammal animal)
!(ecan-fact Inheritance cat mammal)
!(ecan-fact Inheritance oak tree)
!(ecan-add wolf)

; Put dog/wolf in one embedding cluster, oak/tree in another.
; Without this, hashed n-grams will not treat dog and wolf as neighbors.
!(ecan-cluster (dog wolf mammal) (oak tree))

; --- empty working memory ------------------------------------------------
; Focus is empty. STI on dog/wolf/oak should be ~0.
!(ecan-focus)
!(ecan-sti dog)
!(ecan-sti wolf)
!(ecan-sti oak)

; --- pay attention to dog ------------------------------------------------
; Stimulus is a bank transfer. After one cycle, focus should include dog
; and stuff linked to it. wolf can pick up STI via the neural cluster;
; oak should stay near 0.
!(ecan-stimulate dog 12)
!(ecan-sti dog)
!(ecan-cycle)
!(ecan-focus)
!(neural-similar dog)
!(ecan-sti wolf)
!(ecan-sti oak)

; --- infer only on the focused subgraph ---------------------------------
; Should derive (Inheritance dog animal), not anything about oak.
!(ecan-infer)
!(ecan-focus)
