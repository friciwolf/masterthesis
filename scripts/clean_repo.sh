#!/usr/bin/env bash

action() {
    local base="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && /bin/pwd )"
    local origin="$( /bin/pwd )"

    local project_dir="$( dirname "$base" )"
    cd "$project_dir"

    rm -rf \
        *.aux \
        *.bbl \
        *.bcf \
        *.blg \
        *.eps \
        *.fdb_latexmk \
        *.fls \
        *.fls \
        *.gz \
        *.ilg \
        *.lof \
        *.log \
        *.lot \
        *.mp \
        *.nlo \
        *.nls \
        *.out \
        *.t1 \
        *.1 \
        *.dvi \
        *.tar.gz \
        *.tgz \
        *.toc

    cd "$origin"
}
action "$@"
