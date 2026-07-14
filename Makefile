PYTHON ?= .venv/bin/python
CHECKPOINT ?= models/tshape_zero_plus_release.pt

.PHONY: smoke assets verify serve release

smoke:
	TSHAPE_EASYTSAD_ONLY=1 TSHAPE_EASYTSAD_BACKEND=compatible \
		$(PYTHON) experiments/verify_easytsad_protocol.py
	$(PYTHON) experiments/tshape_zero_product.py score \
		--checkpoint $(CHECKPOINT) \
		--values "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,40,17,18,19,20,21,22,23,24,25,10,27,28,29,30,31"

assets:
	$(PYTHON) experiments/build_easytsad_submission_results.py
	$(PYTHON) experiments/make_easytsad_submission_assets.py
	$(PYTHON) experiments/export_strict_submission_results.py
	$(PYTHON) experiments/make_strict_zero_submission_assets.py

verify:
	TSHAPE_EASYTSAD_ONLY=1 TSHAPE_EASYTSAD_BACKEND=compatible \
		$(PYTHON) experiments/verify_strict_release.py \
		--json Results/RENE/strict_release_verification.json

serve:
	$(PYTHON) experiments/tshape_zero_product.py serve \
		--checkpoint $(CHECKPOINT) --host 127.0.0.1 --port 8787

release:
	$(PYTHON) experiments/build_rene_release.py
