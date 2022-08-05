#!/usr/bin/env bash

action() {
    local text_file="$1"

    # standard pdflatex to create the *.mp files
    pdflatex -interaction=nonstopmode "$text_file"

    # mpost for all *.mp files
    for f in *.mp; do
        echo -e "\n\n\n" | mpost "$f";
    done

    # run pdflatex again to include created feynman graphs
    pdflatex -interaction=nonstopmode "$text_file"
}
action "$@"
