# Schizophrenia symptom extraction - NLP pipeline

### Scripts
* `scripts/LMM_inference/start_script.batch` - script used to initiate `symptom_extraction.py` or `symptom_extraction_fs.py` using slurm job management system. This should be run as `sbatch start_script.batch model_name out_dir` where `model_name` is the LLM that you want to use for inference and `out_dir` is the desired name of the output directory. The `--gpus` and `--time` arguments should be changed dependent on which LLM is used. `HF_TOKEN` should be replaced with the users own huggingface token. Miniforge3 was used to create the virtual environment for this project.  
* `scripts/LLM_inference/symptom_extraction.py` - python script that uses vLLM to run inference. Zero-shot prompts are created in a loop using `ratings_guide_global.csv` and these are then passed to the `llm.generate` function using recommended sampling parameters (`sampling_params`). Results are then saved to the `out_dir/cogs_predict_sympt.csv`.
* `scripts/LLM_inference/symptom_extraction_fs.py` - this script follows that in `symptom_extraction.py` but generates and runs inference using few-shot prompts instead (from the `ratings_guide_global_fs.csv`). The script varies primarily where prompts are created, with examples of case notes with a given symptom (or without) included in-prompt.
* `dependencies` - dependencies for running python scripts in the `LLM_inference` folder.

### Data
* `data/ratings_guide_global.csv` - ratings guide used in the `symptom_extraction.py` script to add information regarding symptoms and how they should be defined to the zero-shot LLM prompt.
* `data/ratings_guide_global_fs.csv` - ratings guide used in the `symptom_extraction_fs.py` script to add information regarding symptoms and how they should be defined to the few-shot LLM prompt. Format of the file remains the same as used in the study however, columnd `abrex_1` to `abrex_5` have had case note excerpts removed. Columns `abrat_1` to `abrat_5` are to be populated with either 1 or 0 dependent on the example supplied. For example, for hallucinations, `abrat_1` should be rated 1 if `abrex_1` contains evidence of hallucinations within the text excerpt, and 0 if it contains no evidence of hallucinations (and so on for remaining symptoms).  
* `data/1_CardiffCOGS_records.csv` - example data emulating `1_CardiffCOGS_records` used in `symptom_extraction.py` and `symptom_extraction_fs.py` scripts. Instead of CardiffCOGS dataset, file contains fake study IDs (`Study_ID` column) and example case summaries (`Case_note_summary` column). 

### Analysis