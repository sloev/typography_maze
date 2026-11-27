# Labyrinthography: analyse every book, compare them, then typeset the book.
BOOKS := $(wildcard books/*.yaml)
NUMBERS := $(patsubst books/%.yaml,build/%/numbers.tex,$(BOOKS))

all: book/main.pdf

build/%/numbers.tex: books/%.yaml labyrinth/*.py
	python3 -m labyrinth.build $<

build/the-trial/variants/numbers.tex: books/the-trial.yaml labyrinth/*.py
	python3 -m labyrinth.experiments $<

build/compare/numbers.tex: $(NUMBERS) labyrinth/*.py
	python3 -m labyrinth.compare $(BOOKS)

book/main.pdf: $(NUMBERS) build/the-trial/variants/numbers.tex build/compare/numbers.tex book/*.tex book/labyrinth.sty
	cd book && latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex

zine: zine/zine.pdf zine/zine-print.pdf

build/zine/first-steps.tex: zine/figures.py books/the-trial.yaml labyrinth/*.py
	python3 zine/figures.py

zine/zine.pdf: zine/zine.tex build/zine/first-steps.tex build/compare/numbers.tex build/the-trial/numbers.tex
	cd zine && latexmk -lualatex -interaction=nonstopmode -halt-on-error zine.tex

zine/zine-print.pdf: zine/zine.pdf zine/zine-print.tex
	cd zine && latexmk -lualatex -interaction=nonstopmode -halt-on-error zine-print.tex

clean:
	rm -rf build
	cd book && latexmk -C

.PHONY: all clean zine
