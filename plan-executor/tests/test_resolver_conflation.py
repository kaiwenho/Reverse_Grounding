"""Gene/protein conflation in entity resolution.

Why this file exists at all: the conflation machinery shipped months ago and
was never once exercised. `conflation_from_plan` reads three places on the
plan, and plan-core's schema forbids all three, so the flags were reachable
only from the command line and no caller passed them. The feature was dead,
and nothing said so until a live question lost its answer to it.

The case that surfaced it, question 6 of the final run:

    "Which drugs decrease the activity of TNF in rheumatoid arthritis?"

The planner wrote `biolink_category: Protein` for TNF, because "activity"
reads as a protein. The Name Resolver returned twenty records: the TNF gene,
and nineteen *other* proteins — TNF receptor 14, C1q-and-TNF-related 4, TNF
receptor associated factor 7. The human TNF protein, UniProtKB:P01375, was
not among them. Told to expect a Protein, the model refused every candidate,
which is what its prompt asks of it:

    "No candidate represents the human TNF protein (UniProt P01375); the
     listed entries are a gene or unrelated TNF family members"

The model was right. The category was the problem. These tests pin the fix:
a Gene or Protein entity gets the group by default, so both molecular forms
reach the model and the model decides.

No network, no LLM: `mock_lookup` stands in for the Name Resolver and a small
recording stub stands in for the model.
"""

from types import SimpleNamespace

import pytest

from plan_executor.resolver import DisambiguationChoice, EntityResolver


# --- doubles ---------------------------------------------------------------

# `mock_lookup` replaces the HTTP call, so it yields raw records rather than
# `Candidate`s — the same shape the Name Resolver returns.
TNF_GENE = {
    "curie": "NCBIGene:7124", "label": "TNF",
    "types": ["biolink:Gene", "biolink:GeneOrGeneProduct"],
}
TNF_RECEPTOR = {
    "curie": "UniProtKB:F6Q0M4",
    "label": "TNF receptor superfamily member 14",
    "types": ["biolink:Protein", "biolink:GeneProductMixin"],
}
RHEUMATOID_ARTHRITIS = {
    "curie": "MONDO:0008383", "label": "rheumatoid arthritis",
    "types": ["biolink:Disease"],
}

QUESTION = "Which drugs decrease the activity of TNF in rheumatoid arthritis?"


def entity(entity_ref, name, category, **over):
    fields = dict(
        entity_ref=entity_ref, name=name, biolink_category=category,
        is_variable=False, aliases=[], taxa=[], constraints=[],
        input_binding=None, notes=None,
    )
    fields.update(over)
    return SimpleNamespace(**fields)


class RecordingLookup:
    """The Name Resolver, remembering the type filter it was sent.

    The filter matters as much as the result: a server-side `biolink_type`
    narrows the pool before anything local can widen it, so conflation has to
    suppress it rather than compensate afterwards.
    """

    def __init__(self, *records):
        self.records = list(records)
        self.calls = []

    def __call__(self, query, biolink_type=None):
        self.calls.append(biolink_type)
        if biolink_type is None:
            return list(self.records)
        return [r for r in self.records if biolink_type in r["types"]]


class RecordingModel:
    """The disambiguator, remembering what it was shown, and taking the first.

    Taking index 0 unconditionally is the point: these tests are about which
    candidates reach the model and what it is told about them, not about how
    well it judges. Judgement is the model's business and is not pinned here.
    """

    def __init__(self):
        self.pool = None
        self.conflation = None
        self.expected_category = None

    def choose(self, question, entity_name, expected_category, candidates,
               aliases=(), conflation=()):
        self.pool = [c.curie for c in candidates]
        self.conflation = list(conflation)
        self.expected_category = expected_category
        return DisambiguationChoice(index=0, reason="stub", confidence="high")


def resolve(ent, *records, conflation=(), **resolver_kwargs):
    """Resolve one entity.

    `conflation` is the per-call argument `resolve_plan` computes and hands
    down. The constructor's `conflation=` is a different thing — an override
    read only inside `resolve_plan` — so it is exercised separately, through
    `resolve_plan`, at the bottom of this file.
    """
    lookup, model = RecordingLookup(*records), RecordingModel()
    resolver = EntityResolver(
        disambiguator=model, mock_lookup=lookup, verbose=False,
        **resolver_kwargs,
    )
    res = resolver.resolve_entity(ent, question=QUESTION, conflation=conflation)
    return res, lookup, model


# --- the default fires -----------------------------------------------------

@pytest.mark.parametrize("category", ["Protein", "Gene", "biolink:Protein"])
def test_gene_protein_entity_gets_the_group_by_default(category):
    res, lookup, model = resolve(
        entity("target_tnf", "TNF", category), TNF_GENE, TNF_RECEPTOR,
    )

    # No server-side filter: both molecular forms have to come back.
    assert lookup.calls == [None]
    assert model.pool == ["NCBIGene:7124", "UniProtKB:F6Q0M4"]
    # And the model is told they count as one concept, or it will reject the
    # form that does not match the stated category — which is what happened.
    assert model.conflation == ["gene_protein"]
    assert res.conflation == ["gene_protein"]


def test_the_q6_candidate_pool_now_resolves():
    """The regression itself: a gene reaches the model under a Protein plan."""
    res, _, _ = resolve(
        entity("target_tnf", "TNF", "Protein"), TNF_GENE, TNF_RECEPTOR,
    )
    assert res.curies == ["NCBIGene:7124"]
    assert res.method == "llm"


# --- and stays off everywhere else -----------------------------------------

def test_a_disease_entity_is_untouched():
    """The default is per-entity. A disease anchor in the same plan is not
    a gene or a protein, so nothing about its lookup may change."""
    res, lookup, model = resolve(
        entity("disease_ra", "rheumatoid arthritis", "Disease"),
        RHEUMATOID_ARTHRITIS, TNF_GENE,
    )
    assert lookup.calls == ["biolink:Disease"]
    assert model.pool == ["MONDO:0008383"]
    assert res.conflation == []


def test_drug_chemical_is_not_defaulted_on():
    """Only `gene_protein` is automatic. `drug_chemical` has not been shown
    to be needed, and two behaviour changes at once cannot be told apart."""
    assert list(EntityResolver.DEFAULT_CONFLATION) == ["gene_protein"]
    res, lookup, _ = resolve(
        entity("drug_x", "aspirin", "Drug"),
        {"curie": "CHEBI:15365", "label": "aspirin", "types": ["biolink:Drug"]},
    )
    assert lookup.calls == ["biolink:Drug"]
    assert res.conflation == []


def test_an_uncategorised_entity_gets_nothing():
    """With no category there is no group to belong to, and the lookup is
    already unfiltered."""
    res, lookup, _ = resolve(entity("thing", "TNF", ""), TNF_GENE)
    assert lookup.calls == [None]
    assert res.conflation == []


# --- the escape hatch ------------------------------------------------------

def test_default_conflation_false_restores_the_old_behaviour():
    """`--no-default-conflation`. A run made before this default existed has
    to stay reproducible, or the earlier traces cannot be re-derived."""
    res, lookup, model = resolve(
        entity("target_tnf", "TNF", "Protein"), TNF_GENE, TNF_RECEPTOR,
        default_conflation=False,
    )
    assert lookup.calls == ["biolink:Protein"]
    assert model.pool == ["UniProtKB:F6Q0M4"]   # the gene never reaches it
    assert res.conflation == []


def test_an_explicit_flag_is_not_duplicated():
    """`--conflate gene_protein` and the default ask for the same thing."""
    res, _, model = resolve(
        entity("target_tnf", "TNF", "Gene"), TNF_GENE, TNF_RECEPTOR,
        conflation=["gene_protein"],
    )
    assert res.conflation == ["gene_protein"]
    assert model.conflation == ["gene_protein"]


def test_an_explicit_unrelated_flag_is_kept_alongside_the_default():
    res, _, _ = resolve(
        entity("target_tnf", "TNF", "Protein"), TNF_GENE, TNF_RECEPTOR,
        conflation=["drug_chemical"],
    )
    assert res.conflation == ["drug_chemical", "gene_protein"]


def test_resolve_plan_threads_the_constructor_override_down():
    """`--conflate` reaches an entity only through `resolve_plan`, which asks
    `conflation_from_plan` for the flags. Nothing else reads the override, so
    a test that calls `resolve_entity` directly cannot see it — which is how
    an override test can pass while proving nothing."""
    lookup, model = RecordingLookup(TNF_GENE, TNF_RECEPTOR), RecordingModel()
    resolver = EntityResolver(
        disambiguator=model, mock_lookup=lookup, verbose=False,
        conflation=["drug_chemical"], default_conflation=False,
    )
    plan = SimpleNamespace(question=QUESTION, raw={})
    out = resolver.resolve_plan(
        plan, entities={"target_tnf": entity("target_tnf", "TNF", "Protein")},
    )
    assert out["target_tnf"].conflation == ["drug_chemical"]


# --- the ledger tells the truth --------------------------------------------

def test_a_variable_entity_records_no_flag_it_never_used():
    """Variable entities are left open by category and never looked up.
    Recording a flag that had no effect would put a claim in the ledger that
    nothing acted on."""
    res, lookup, _ = resolve(entity("drug_v", "", "Protein", is_variable=True))
    assert res.method == "variable"
    assert res.conflation == []
    assert lookup.calls == []


def test_a_supplied_curie_records_no_flag_it_never_used():
    """A CURIE stated in the plan is the author's decision; no lookup and no
    disambiguation run, so no conflation applied to it."""
    ent = entity("target_tnf", "TNF", "Protein", curies=["NCBIGene:7124"])
    res, lookup, _ = resolve(ent, TNF_GENE)
    assert res.method == "supplied"
    assert res.conflation == []
    assert lookup.calls == []


def test_the_ledger_states_every_category_that_was_accepted():
    """The loop controller cannot recompute this — it has neither the flags
    nor the group table — so the resolver states it.

    Without the field, the controller compared a resolved Gene against a plan
    that said Protein, called it a mismatched anchor, and repaired a plan that
    had already returned eight correct candidates.
    """
    res, _, _ = resolve(
        entity("target_tnf", "TNF", "Protein"), TNF_GENE, TNF_RECEPTOR,
    )
    accepted = res.to_dict()["accepted_categories"]
    assert "Gene" in accepted and "Protein" in accepted
    assert accepted == sorted(accepted)   # stable across runs


def test_without_conflation_only_the_plans_category_is_accepted():
    res, _, _ = resolve(
        entity("disease_ra", "rheumatoid arthritis", "Disease"),
        RHEUMATOID_ARTHRITIS,
    )
    assert res.accepted_categories == ["Disease"]


def test_an_entity_that_was_never_looked_up_accepts_nothing():
    """Variable and plan-supplied entities skip resolution entirely. An
    accepted set there would describe a comparison that never happened."""
    var = entity("drug_v", "", "SmallMolecule", is_variable=True)
    res, _, _ = resolve(var)
    assert res.accepted_categories == []


def test_the_ledger_reports_the_flags_that_actually_applied():
    """`conflation` in the resolution ledger is the audit trail for this
    default — it is the only place a reader can see that the pool was widened
    without being asked."""
    res, _, _ = resolve(
        entity("target_tnf", "TNF", "Protein"), TNF_GENE, TNF_RECEPTOR,
    )
    assert res.to_dict()["conflation"] == ["gene_protein"]
