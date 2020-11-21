for d in *.md; do name=$(echo "$d" | cut -f 1 -d '.'); pandoc $d --pdf-engine=xelatex -o $name.pdf; done
