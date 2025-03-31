PYTHON = python

.PHONY: all
all: install test

.PHONY: install
install:
	pip install .

.PHONY: test
test:
	$(PYTHON) -m unittest discover -s tests -v --failfast

.PHONY: clean
clean:
	rm -f *.csv *.end *.mtr *.mdd *.bnd *.dxf *.audit *.err *.eso *.json *.rdd *.shd *.sql *.htm *.eio *.mtd
