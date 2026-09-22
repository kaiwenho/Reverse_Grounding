# Reverse Grounding
#
# The packages are siblings that depend on each other by name, so the order in
# PACKAGES matters: plan-core first, then anything that imports it.
#
# Everything here assumes editable installs into one environment. That is not
# only convenience — the loop-controller runs the executor as a *subprocess*,
# and with editable installs that subprocess inherits the environment and
# finds plan_executor on its own. Without them every command needs a
# PYTHONPATH and two --executor-pythonpath flags, which is how this repo used
# to be run and the single most common way to get it wrong.

PY       ?= python3
PIP      ?= $(PY) -m pip
PACKAGES := plan-core planner plan-executor loop-controller query-intake

# The Biolink Model is a version-pinned download, not a source file, so it is
# gitignored and a fresh clone does not have it. Nothing works without it: the
# schema compiles its category and qualifier enums from this YAML, so a clone
# that skips this step fails in the planner with a traceback that never
# mentions Biolink. Hence `install` and `test` both depend on it.
BIOLINK_VERSION ?= v4.4.3
BIOLINK_YAML    := plan-core/data/biolink-model.yaml
BIOLINK_URL     := https://raw.githubusercontent.com/biolink/biolink-model/$(BIOLINK_VERSION)/biolink-model.yaml

.DEFAULT_GOAL := help
.PHONY: help install install-dev test probe biolink clean

help:
	@echo "Reverse Grounding"
	@echo
	@echo "  make install      the Biolink Model, then editable installs of all five packages"
	@echo "  make install-dev  the above, plus pytest"
	@echo "  make test         every suite; no model needed"
	@echo "  make biolink      just fetch the pinned Biolink Model"
	@echo "  make probe        the relaxation probe (needs ARAX and a model)"
	@echo "  make clean        remove __pycache__ and build artefacts"
	@echo
	@echo "To ask a question, use the command itself:"
	@echo "  reverse-grounding --question \"Which drugs treat psoriasis?\""

biolink: $(BIOLINK_YAML)

$(BIOLINK_YAML):
	@echo "==> Biolink Model $(BIOLINK_VERSION)"
	@curl -fsSL "$(BIOLINK_URL)" -o "$@.part" \
		&& mv "$@.part" "$@" \
		|| { rm -f "$@.part"; \
		     echo "could not download $(BIOLINK_URL)"; \
		     echo "see plan-core/data/README.md to fetch it by hand"; \
		     exit 1; }

install: biolink
	@for p in $(PACKAGES); do \
		echo "==> $$p"; \
		$(PIP) install -e "$$p" || exit 1; \
	done
	@echo
	@echo "Installed. 'make test' to check."
	@echo "Then: reverse-grounding --question \"Which drugs treat psoriasis?\""

install-dev: install
	@$(PIP) install "pytest>=7.0"

# plan-core has no tests yet; the loop skips a package with no tests directory
# rather than failing, so adding one needs no change here.
test: biolink
	@failed=""; \
	for p in $(PACKAGES); do \
		if [ ! -d "$$p/tests" ]; then \
			echo "==> $$p (no tests)"; \
			continue; \
		fi; \
		echo "==> $$p"; \
		( cd "$$p" && $(PY) -m pytest -q ) || failed="$$failed $$p"; \
	done; \
	if [ -n "$$failed" ]; then echo; echo "FAILED:$$failed"; exit 1; fi; \
	echo; echo "All suites passed."

# There is deliberately no `demo` target. A Makefile earns its place by hiding
# something hard — the install order, the Biolink download, the per-package
# test invocation. `reverse-grounding --question "..."` is already one plain
# command, so wrapping it would add a name to learn without removing one.
#
# The one plan known to reach `relax_plan` against the live graph. Takes a few
# minutes. tools/probes/README.md explains what to read afterwards.
probe:
	@echo "probe" > /tmp/rg-probe.txt
	cd loop-controller && $(PY) tools/bench.py /tmp/rg-probe.txt \
		--plan tools/probes/probe_predicate.json \
		--out runs/probe-predicate

clean:
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name '*.egg-info' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name .pytest_cache -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned"
