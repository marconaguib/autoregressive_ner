import datetime
import hashlib
import itertools
import json
import os
import re
import datasets
import numpy as np
from sklearn.model_selection import KFold
import argparse
import logging
import random

import torch
from bloom_predict import bloom_predict
from fastchat.model import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from nlstruct import BRATDataset, HuggingfaceNERDataset
from nlstruct.metrics import MetricsCollection
from nlstruct.registry import get_instance



args = argparse.ArgumentParser()
args.add_argument("--language", type=str, default="fr", help="language of the dataset")
args.add_argument("--domain", type=str, default="general", help="domain of the dataset")
args.add_argument("--ner_tag", type=str, help="ner tag to evaluate")
args.add_argument("--begin_tag", type=str, default="@@")
args.add_argument("--end_tag", type=str, default="##")
args.add_argument("--n_few_shot", type=int, default=5)
args.add_argument("--model_name", type=str, default="bigscience/bloom")
args.add_argument("--batch_size", type=int, default=2)
args.add_argument("--criterion", type=str, default="most_occurences")
args.add_argument("--prompt_dict", type=str)
args.add_argument('--top_p', type=float, default=1.0)
args.add_argument('--top_k', type=int, default=50)
args.add_argument('--temperature', type=float, default=1.0)
args.add_argument('--num_beams', type=int, default=1)
args.add_argument('--api_inference', action="store_true")
args.add_argument('--random_seed_acquisition', type=int, default=1)
args.add_argument('--random_seed_prompt_generation', type=int, default=42)
args.add_argument('-s', '--training_size', type=int, default=70)
args.add_argument('-t', '--test_on_test_set', action="store_true")
args.add_argument('--do_sample', action="store_true")
args.add_argument('--no_control', dest='control', action='store_false')
args.add_argument('--no_self_verification', dest='self_verification', action='store_false')
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloom_ner")

#random deals with choosing the few-shot examples, so we want that fixed
random.seed(args.random_seed_prompt_generation)

prompt_keywords = {
    'en' : {
        'first_sentence' : "I am an excellent {}. The task is to label all mentions of {} in a sentence. {} I can also put them in a specific format. Here are some examples of sentences I can handle:\n",
        'last_sentence' : "Imitate me. Identify all the mentions of {} in the following sentence, by putting \"{}\" in front and a \"{}\" behind each of them.\n",
        'domains_jobs' : {
            'clinical' : "clinician",
            'general' : "linguist"
            },
        'ner_tags_plural' : {
            'PER' : "person names",
            'DISO' : "disorders",
            'LOC' : "places",
            'ORG' : "organizations",
            'ANAT' : "parts of the body",
            "LIVB" : "living beings",
            "PROC" : "procedures",
            "FAC" : "facilities",
            },
        'ner_tags' : {
            'PER' : "a person's name",
            'DISO' : "an alteration of the functions of the body",
            'LOC' : "a place",
            'ORG' : "an organization",
            'ANAT' : "a part of the body",
            "LIVB" : "a living being",
            "PROC" : "a procedure",
            "FAC" : "a facility",
            },
        'ner_tags_description' : {
            'PER' : "These are words that refer to the name of a real or fictional person.",
            'DISO' : "These are words that refer to an alteration or abnormality of the functions or health of the body.",
            'LOC' : "These are words that refer to the name of a place.",
            'ORG' : "These are words that refer to the name of an organization.",
            'ANAT' : "These are words that refer to a part of the human body.",
            "LIVB" : "These are words that refer to a living being.",
            "PROC" : "These are words that refer to a medical procedure.",
            "FAC" : "These are words that refer to a facility made by humans.",
            },
        'input_intro' : "Input: ",
        'output_intro' : "Output: ",
        'first_sentence_self_verif' : "I am an excellent {}. The task is to verify whether a given word is a mention of a {}. Below some examples :\n",
        "self_verif_template": "In the sentence \"{sentence}\", is \"{{word}}\" {ner_tag}?\n",
        "yes": "Yes",
        "no": "No",
        }
    ,
    'vicuna_assistant' : {
        'first_sentence' : "A chat between a curious {} and an artificial intelligence assistant. The assistant can label all mentions of {} in a sentence. {} It can also put them in a specific format. Here are some examples of sentences it can handle:\n",
        'last_sentence' : "",
        'domains_jobs' : {
            'clinical' : "clinician",
            'general' : "linguist"
            },
        'ner_tags_plural' : {
            'PER' : "person names",
            'DISO' : "disorders",
            'LOC' : "places",
            'ORG' : "organizations",
            'ANAT' : "parts of the body",
            'LIVB' : "living beings",
            'PROC' : "procedures",
            },
        'ner_tags' : {
            'PER' : "a person's name",
            'DISO' : "an alteration of the functions of the body",
            'LOC' : "a place",
            'ORG' : "an organization",
            'ANAT' : "a part of the body",
            'LIVB' : "a living being",
            'PROC' : "a procedure",
            },
        'ner_tags_description' : {
            'PER' : "These are words that refer to the name of a real or fictional person.",
            'DISO' : "These are words that refer to an alteration or abnormality of the functions or health of the body.",
            'LOC' : "These are words that refer to the name of a place.",
            'ORG' : "These are words that refer to the name of an organization.",
            'ANAT' : "These are words that refer to a part of the human body.",
            'LIVB' : "These are words that refer to a living being.",
            'PROC' : "These are words that refer to a medical procedure.",
            },
            'input_intro' : "USER : ",
            'output_intro' : "ASSISTANT : ",
            'first_sentence_self_verif' : "A chat between a curious {} and an artificial intelligence assistant. The assistant can verify whether a given word is a mention of a {}. Below some examples :\n",
            "self_verif_template": "USER : In the sentence \"{sentence}\", is \"{{word}}\" {ner_tag}?\n",
            "yes": "ASSISTANT : Yes",
            "no": "ASSISTANT : No",
    },
    'fr' : {
        'first_sentence' : "Je suis un {} expert, je sais identifier les mentions des {} dans une phrase. {} Je peux aussi les mettre en forme. Voici quelques exemples de phrases que je peux traiter :\n",
        'last_sentence' : "Imite-moi. Identifie les mentions de {} dans la phrase suivante, en mettant \"{}\" devant et un \"{}\" derrière la mention dans la phrase suivante.\n",
        'domains_jobs' : {
            'clinical' : "clinicien",
            'general' : "linguiste"
        },
        'ner_tags_plural' : {
            'PER' : "noms de personnes",
            'DISO' : "maladies et symptômes",
            'LOC' : "lieux",
            'ORG' : "organisations",
            'ANAT' : "parties du corps",
            'LIVB' : "êtres vivants",
            'PROC' : "procédures médicales",
            "FAC" : "installations",
        },
        'ner_tags' : {
            'PER' : "un nom de personne",
            'DISO' : "une altération des fonctions du corps",
            'LOC' : "lieu",
            'ORG' : "une organisation",
            'ANAT' : "une partie du corps",
            'LIVB' : "un être vivant",
            'PROC' : "une procédure médicale",
            "FAC" : "une installation",
        },
        'ner_tags_description' : {
            'PER' : "Il s'agit des mots faisant mention du nom d'un personne qu'elle soit réelle ou fictive.",
            'DISO' : "Il s'agit des mots faisant mention d'une altération ou une anormalité des fonctions ou de la santé du corps.",
            'LOC' : "Il s'agit des mots faisant mention du nom d'un lieu.",
            'ORG' : "Il s'agit des mots faisant mention du nom d'une organisation.",
            'ANAT' : "Il s'agit des mots faisant mention d'une partie du corps humain.",
            'LIVB' : "Il s'agit des mots faisant mention d'un être vivant.",
            'PROC' : "Il s'agit des mots faisant mention d'une procédure médicale.",
            "FAC" : "Il s'agit des mots faisant mention d'une installation faite/construite par les humains.",
        },
        'input_intro' : "Entrée : ",
        'output_intro' : "Sortie : ",
        'first_sentence_self_verif' : "Je suis un {} expert, je sais identifier si un mot est une mention des {} dans une phrase. Voici quelques exemples de phrases que je peux traiter :\n",
        "self_verif_template": "Dans la phrase \"{sentence}\", le mot \"{{word}}\" désigne-t-il {ner_tag} ?\n",
        "yes": "Oui",
        "no": "Non",
    }
}

if args.domain == 'general':
    dataset = HuggingfaceNERDataset(
        dataset_name='meczifho/WikiNER',
        subset=args.language,
        tag_map={
            0: "O",
            1: "LOC",
            2: "PER",
            3: "FAC",
            4: "ORG",
        },
        doc_id_colname="id",
    )
    ner_tags = ['PER', 'LOC', 'ORG', 'FAC']
else :
    dataset = BRATDataset(
        train= "/mnt/beegfs/home/naguib/autoregressive_ner/quaero/training",
        val= 0, 
        test= "/mnt/beegfs/home/naguib/autoregressive_ner/quaero/test",
    )
    ner_tags = ['DISO', 'ANAT', 'PROC', 'LIVB']

if not args.training_size:
    raise ValueError("Please specify training size")
if not args.prompt_dict:
    raise ValueError("Please specify prompt dictionary")

traindev_dataset = [e for e in dataset.train_data if len(e['text']) < 512]
test_dataset = [e for e in dataset.test_data if len(e['text']) < 512]

time_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = 'hyp_search_'+time_date
os.mkdir(folder_name)

logger.info("Loading model...")
model, tokenizer = load_model(
        args.model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_gpus=1,
        load_8bit='vicuna' in args.model_name,
        debug=False,
        )
#np random deals with choosing the traindev dataset
np.random.seed(args.random_seed_acquisition)
traindev_dataset_this_seed = [traindev_dataset[i] for i in np.random.choice(len(traindev_dataset), size=args.training_size, replace=False)]

results = {}

time_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logfile = open(folder_name+'/log_'+time_date+'.txt','w')
logfile.write('language: '+args.language+'\n')
logfile.write('domain: '+args.domain+'\n')
logfile.write('begin_tag: '+args.begin_tag+'\n')
logfile.write('end_tag: '+args.end_tag+'\n')
logfile.write('n_few_shot: '+str(args.n_few_shot)+'\n')
logfile.write('model_name: '+args.model_name+'\n')
logfile.write('criterion: '+args.criterion+'\n')
logfile.write('prompt_dict: '+args.prompt_dict+'\n')
logfile.write('training_size: '+str(args.training_size)+'\n')
logfile.write('random_seed: '+str(args.random_seed_acquisition)+'\n')
logfile.write('control: '+str(args.control)+'\n')
logfile.write('num_beams: '+str(args.num_beams)+'\n')
logfile.write('self verification: '+str(args.self_verification)+'\n')
# logfile.write('example prompt: \n'+prompts[0]+'\n')
# logfile.write('self_verif_template: \n'+self_verif_template+'\n')
if args.do_sample:
    logfile.write('top_p: '+str(args.top_p)+'\n')
    logfile.write('top_k: '+str(args.top_k)+'\n')
    logfile.write('temperature: '+str(args.temperature)+'\n')
else:
    logfile.write('greedy'+'\n')
logfile.write('='*50+'\n')

textual_outputs, predicted_dataset = bloom_predict(
    training_data=traindev_dataset_this_seed,
    testing_data=test_dataset if args.test_on_test_set else None,
    ner_tags=ner_tags,
    model_name=args.model_name,
    logger=logger,
    begin_tag=args.begin_tag,
    end_tag=args.end_tag,
    self_verification=args.self_verification,
    model=model,
    tokenizer=tokenizer,
    control=args.control,
    n_few_shot=args.n_few_shot,
    criterion=args.criterion,
    keywords=prompt_keywords[args.prompt_dict],
    domain=args.domain,
    model_kwargs={
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
    }
)

logger.info("Evaluating...")
metric_names = {
        "exact": dict(module="dem", binarize_tag_threshold=1., binarize_label_threshold=1., add_label_specific_metrics=ner_tags),
        "partial": dict(module="dem", binarize_tag_threshold=1e-5, binarize_label_threshold=1., add_label_specific_metrics=ner_tags),
}
metrics = MetricsCollection({k: get_instance(m) for k, m in metric_names.items()})

for metric in metrics.values():
        metric(predicted_dataset, test_dataset if args.test_on_test_set else traindev_dataset_this_seed)
        print(metric.compute())
        logfile.write(str(metric.compute())+'\n')

# for o, pred, gold in zip(textual_outputs, predicted_dataset, test_dataset if args.test_on_test_set else traindev_dataset_this_seed):
#     logfile.write('='*50+'\n')
#     logfile.write('input: '+pred['text']+'\n')
#     logfile.write('output: '+o+'\n')
#     logfile.write('final: '+str([p['text'] for p in pred['entities']])+'\n')
#     logfile.write('gold: '+str([g['text'] for g in gold['entities'] if g['label']=='PER'])+'\n')
for i, (o, pred, gold) in enumerate(zip(textual_outputs, predicted_dataset, test_dataset if args.test_on_test_set else traindev_dataset_this_seed)):
        logfile.write('='*50+'\n')
        logfile.write('input: '+pred['text']+'\n')
        logfile.write('-'*50+'\n')
        for j,tag in ner_tags:
            logfile.write(tag+' output: '+textual_outputs[i+len(textual_outputs)*j]+'\n')
            logfile.write('final: '+str([p['text'] for p in pred['entities'] if p['label']==tag])+'\n')
            logfile.write('gold: '+str([g['text'] for g in gold['entities'] if g['label']==tag])+'\n')

logfile.close()