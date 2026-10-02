from graphrag.extract import extract_from_text, entity_id


def test_extracts_acquisition_and_ceo():
    text = (
        "Google acquired DeepMind in 2014. Sundar Pichai is the CEO of Google. "
        "DeepMind was founded by Demis Hassabis in London."
    )
    extracted = extract_from_text(text)
    types = {(r.source_id, r.type, r.target_id) for r in extracted.relations}
    assert (entity_id("Google"), "ACQUIRED", entity_id("DeepMind")) in types
    assert (entity_id("Sundar Pichai"), "CEO_OF", entity_id("Google")) in types
    assert (entity_id("Demis Hassabis"), "FOUNDED", entity_id("DeepMind")) in types
    assert (entity_id("DeepMind"), "LOCATED_IN", entity_id("London")) in types
