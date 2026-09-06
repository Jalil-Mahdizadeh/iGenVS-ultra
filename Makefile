.PHONY: images check dock screen

N ?= 10000

images:
	./scripts/build-images.sh

check:
	./scripts/check-release.sh

dock:
	./igenvs-ultra dock \
		--complex complexes/4ag8.pdb \
		--ligand-id A:AXI:2000 \
		--input iGenVS/examples/library.csv \
		--output-dir runs/example-docking \
		--pose-output none

screen:
	IGENVS_ULTRA_JOB="$(CURDIR)/examples/4ag8-screen" \
		./igenvs-ultra screen-fast "$(N)"
