#! /bin/bash
files=(1_Introduction.ipynb 2-Grain-SMOX.ipynb 3-Resistance-sensor.ipynb 4-Exp-data.ipynb 5-Empirical-Studies.ipynb)
outdir=latex


cp -r media latex/
cp -r ipython.bib latex/

for f in ${files[*]};
do
jupyter-nbconvert --to latex --output-dir $outdir $f
done

cd $outdir

for i in {1..3};
do
	for f in ${files[*]};
	do
		pdflatex "$(basename "$f" .ipynb).tex"
	done
echo $i
done


for f in ${files[*]};
do
	bibtex "$(basename "$f" .ipynb).aux"
done

for f in ${files[*]};
do
	pdflatex   "$(basename "$f" .ipynb).tex"
done

for f in ${files[*]};
do
	cp "$(basename "$f" .ipynb).pdf" "../PDFs"
done

