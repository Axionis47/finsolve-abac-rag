from app.retrieval.filtering import prefilter_by_allowed_roles


def test_prefilter_by_allowed_roles():
    items = [
        {"id": 1, "allowed_roles": ["engineering", "c_level"]},
        {"id": 2, "allowed_roles": ["hr", "c_level"]},
        {"id": 3, "allowed_roles": ["marketing", "employee"]},
        {"id": 4, "allowed_roles": []},
    ]
    eng = prefilter_by_allowed_roles(items, "engineering")
    assert {x["id"] for x in eng} == {1}
    hr = prefilter_by_allowed_roles(items, "hr")
    assert {x["id"] for x in hr} == {2}
    emp = prefilter_by_allowed_roles(items, "employee")
    assert {x["id"] for x in emp} == {3}

