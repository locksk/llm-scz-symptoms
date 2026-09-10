import sys
import os
import json
import glob
import torch
import numpy as np
import pandas as pd
import re
import csv
import gc
import random
import bitsandbytes
from vllm import LLM, SamplingParams
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html

TEST_RUN = True

# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

################################################################################################################################
################################################################################################################################

def clean(txt):
#Clean text by removing extra spaces and special characters
    if txt.startswith(','):
        txt = re.sub(',', '', txt, 1)
    if txt.startswith('.'):
        txt = re.sub('.', '', txt, 1)
    txt = re.sub(r'\s+', ' ',   txt)
    txt = re.sub(r'\--+', '', txt)
    txt = txt.lstrip().replace('<d2>',' ').replace('<c9>', ' ').replace('<d3>',' ').replace('<d5>','').strip()
    txt = txt.rstrip()
    return txt


def readData(fname):
# Read CSV data and clean the second column
    data = pd.read_csv(fname, encoding="utf-8")
    data.iloc[:, 1] = data.iloc[:, 1].astype(str)
    data.iloc[:, 1] = data.iloc[:, 1].apply(np.vectorize(clean))
    return data


def readGuide(fname):
# Read and clean the ratings guide
    data = pd.read_csv(fname, encoding="utf-8")
    data = data.astype(str)
    data = data.apply(np.vectorize(clean))
    return data

#################################################################################################################################
################################################################################################################################

def create_prompt(note, question):
    text = f""" 
    Question: {question}           

    Patient note: {note}
            """
    prompt = [
        {'role': 'system', 'content': f"""You are a world-leading psychiatrist. Analyse the patient note for symptom presence or absence. Double check your answer before responding. Respond with only "1" or "0"."""},
        {'role': 'user', 'content': text}]  
    return prompt


def create_prompt_ds(note, question):
    text = f"""You are a world-leading psychiatrist. Analyse the patient note for symptom presence or absence. Double check your answer before responding. Respond with only "1" or "0".
            {question}
            
            Patient note: {note}

            <think>\n
        """
    prompt = [{'role': 'user', 'content': text}]
    return prompt

################################################################################################################################
################################################################################################################################

def main():
    random.seed(666)
    model_name = sys.argv[1].casefold()
    outdir_name = sys.argv[2]
    # Set up paths
    PATH = f'/scratch/u6cn/locksk.u6cn/llms/res/ZS/{model_name}/{outdir_name}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    download_dir = "/projects/u6cn/public/hf"
    os.system('export HF_HUB_CACHE={download_dir}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(">> Session Info:")
    print("> Model name: ")
    print(model_name)
    print("> Running on: ")
    print(device)
    print("> Number of GPUs available: ")
    print(torch.cuda.device_count())
    print("> Saving output to:")
    print(PATH)
    print()
    print()
    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=download_dir)
    print(f'>> Running symptom extraction using {model_name}')
    print()
    print(">> Setting paths to clinical summaries and rating guide ...")
    print()
    guide = '/scratch/u6cn/locksk.u6cn/llms/data/ratings_guide_global_zs.csv'
    fname = '/scratch/u6cn/locksk.u6cn/llms/data/1_CardiffCOGS_records.csv'

    print(">> Reading files ...")
    print()
    notes = readData(fname)
    rg = readData(guide)
    if TEST_RUN == True:
        notes = notes.head(50) # testing on a subset of the total sample


    print(">> Generating prompts...")
    print()
    prompts = []
    prompt_metadata = []
    for k,note in enumerate(notes.iloc):
        for s in range(rg.shape[0]):
            cogs_id = note['Study_ID']
            symp_name = rg.iloc[s]['Symptom']
            definition = rg.iloc[s]['Description']
            quest = f"""The following is a psychiatric patient note for a symptom classification task. Please scan the text for keywords related to {symp_name} and identify any descriptions matching the definition. {definition}. 
            
            Based on your analysis, is there evidence of {symp_name} at any point throughout the patient note? Respond "1" if there is evidence of {symp_name}, even if it appears in an historical context or is described synonymously (i.e., not explicitly stated by name). Respond "0" if there is no evidence of {symp_name}. Thank you."""
            
            if "deepseek" in model_name: 
                prompt = create_prompt_ds(note['Case_note_summary'], quest)
            else:
                prompt = create_prompt(note['Case_note_summary'], quest)
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(text)
            prompt_metadata.append({'Study_ID': cogs_id, 'Symptom': symp_name})
    
    
    print(">> Loading LLMs ...")
    print()
    
    if "fp8" in model_name: 
        print("> Preparing FP8 quantised model ...")
        print()
        # for models which have already undergone 8-bit quantisation
        llm = LLM(
            model=model_name,
            download_dir = download_dir,
            tensor_parallel_size=torch.cuda.device_count(), 
            dtype=torch.bfloat16,
            trust_remote_code=True,)
    else:
        # for in-flight 4-bit quantisation with bitsandbytes
        print("> Preparing for in-flight 4-bit quantisation with bitsandbytes ...")
        print()
        llm = LLM(
            model=model_name,
            download_dir = download_dir,
            tensor_parallel_size=torch.cuda.device_count(), 
            dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization="bitsandbytes",
            )

    print()
    print()
    print()

    # where available, using recommended sampling parameters based on model cards
    if model_name == 'mistralai/mistral-nemo-instruct-2407':
        sampling_params = SamplingParams(temperature = 0.3, max_tokens=32768)
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()
    elif model_name == 'mistralai/mistral-small-24b-instruct-2501':
        sampling_params = SamplingParams(temperature = 0.15, max_tokens=32768)
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()
    elif model_name == 'mistralai/mistral-large-instruct-2411':
        sampling_params = SamplingParams(temperature = 0.7, max_tokens=32768)
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()
    elif model_name == 'qwen/qwen3-next-80b-a3b-instruct':
        sampling_params = SamplingParams(temperature = 0.7, top_p = 0.8, top_k = 20, min_p = 0, max_tokens=32768)
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()
    elif model_name == 'qwen/qwen3-next-80b-a3b-thinking':
        sampling_params = SamplingParams(temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0, max_tokens=32768) 
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()
    elif model_name == 'qwen/qwen3-235b-a22b-instruct-2507':
        sampling_params = SamplingParams(temperature = 0.7, top_p = 0.8, top_k = 20, min_p = 0, max_tokens=32768)
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()
    elif model_name == 'qwen/qwen3-235b-a22b-thinking-2507-fp8':
        sampling_params = SamplingParams(temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0, max_tokens=32768)
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()
    elif model_name == 'deepseek-ai/deepseek-r1-distill-qwen-32b':
        sampling_params = SamplingParams(temperature = 0.6, max_tokens=32768)
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()
    elif model_name == 'deepseek-ai/deepseek-r1-distill-llama-70b':
        sampling_params = SamplingParams(temperature = 0.6, max_tokens=32768)  
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()  
    elif model_name == 'aaditya/llama3-openbiollm-70b':
        sampling_params = SamplingParams(temperature = 0, top_p = 0.9, truncate_prompt_tokens = 8172) # model was trained with a short max_model_len of 8192, even though llama3 70B is longer    
        print(f'> Using recommended sampling parameters (based on model card). Info: {sampling_params}')
        print()  
    else:
        sampling_params = SamplingParams(max_tokens=32768) 
        print(f'> Using default sampling parameters. Info: {sampling_params}')
        print()
    
    print(">> Generating responses ...")
    print()
    # generate
    outputs = llm.generate(prompts, sampling_params)
    
    print()
    print(">> Outputting results ...")
    print()

    res_path = f'{PATH}/cogs_predict_sympt.csv'

    results = []
    for idx, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        cogs_id =  prompt_metadata[idx]['Study_ID']
        symp = prompt_metadata[idx]['Symptom']
        results.append([
            cogs_id,
            symp,
            prompt,
            generated_text
        ])

    fields = ['Study_ID', "Symptom", 'Prompt', 'Generated_text']
    with open(res_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(results)
    
    
    # Clean up after inference
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # Reset CUDA device to fully clear memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print(f'>> Symptom extraction using {model_name} complete')


################################################################################################################################
################################################################################################################################

if __name__ == "__main__":
    print()
    print(">> Running main...")
    print()
    main()
    print(">> Complete!")
    print()

