# Task 2 methods and results

To work on a new batch of data, I synthesized a new dataset named "synth_diamonds.csv" using the SDV 
Python library, creating a new dataset with the same metadata as the original one. Ingestion
is performed by merging old data with the newly retrieved one.

I avoided complex processing pipelines, and I kept it all local, moving on cloud solutions for
the next tasks.

To automate the pipeline, I included a .bat file that executes for a fixed number of times 
(currently 5).

The model is trained, tested and evaluated against the full dataset, comprised of the given
one and the synthetic one. It is once again tuned using the Optuna library.

All results are reported in the "evaluation_metrics.txt" file.

Right now, the tuning procedure is the bulk of the computational effort. However, it can
be reduced in a later time.