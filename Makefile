# Labyrinthography: build the analysis for every book, then typeset the book.
BOOKS := $(wildcard books/*.yaml)
SLUGS := $(patsubst books/%.yaml,build/%/numbers.tex,$(BOOKS))

all: book/main.pdf

build/%/numbers.tex: books/%.yaml labyrinth/*.py
	python3 -m labyrinth.build $<

book/main.pdf: $(SLUGS) book/*.tex book/labyrinth.sty
	cd book && latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -rf build
	cd book && latexmk -C

.PHONY: all clean
