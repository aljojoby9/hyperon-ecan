from hyperon_ecan import ECAN, ECANParams, attach_clusters, cosine, hashed_embedding


def test_similar_spellings_are_close():
    dog = hashed_embedding("dog")
    dogs = hashed_embedding("dogs")
    invoice = hashed_embedding("invoice")
    assert cosine(dog, dogs) > cosine(dog, invoice)


def test_neural_spreading_without_symbolic_link():
    net = ECAN(
        ECANParams(
            neural_mix=1.0,
            neural_threshold=0.2,
            max_spread=0.5,
            focus_boundary=5.0,
            rent=0.0,
        )
    )
    net.add_many(["dog", "dogs", "invoice"])
    net.stimulate(["dog"], amount=20.0)
    before = net.atoms["dogs"].sti
    net.cycle()
    assert net.atoms["dogs"].sti > before
    assert net.atoms["dogs"].sti > net.atoms["invoice"].sti


def test_query_similar_ranks_neighbors():
    net = ECAN()
    net.add_many(["cat", "cats", "volcano"])
    names = [n for n, _ in net.query_similar("cat", k=2)]
    assert names[0] == "cats"


def test_cluster_embeddings_beat_hash_for_semantics():
    net = ECAN()
    attach_clusters(net, [["dog", "wolf"], ["oak", "invoice"]])
    dog_wolf = cosine(net.atoms["dog"].embedding, net.atoms["wolf"].embedding)
    dog_oak = cosine(net.atoms["dog"].embedding, net.atoms["oak"].embedding)
    assert dog_wolf > 0.7
    assert dog_wolf > dog_oak
