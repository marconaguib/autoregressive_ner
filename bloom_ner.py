import datetime
import hashlib
import os
import random
import re
import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import logging

def example2string(example, ner_tag_id, begin_tag, end_tag, tagged=True):
    # if ner_tag_id = 3 and 3 stands for LOC, beginning tag = @@ and ending tag = ##
    # and the example is {'id': 0, 'words': ['I', 'love', 'Paris', 'and', 'Berlin'], 'ner_tags': [0, 0, 3, 0, 3]}
    # the returned string will be 'I love @@Paris## and @@Berlin##'
    words = example['words' if 'words' in example else 'tokens']
    ner_tags = example['ner_tags']
    string = ''
    for i, (word, ner_tag) in enumerate(zip(words, ner_tags)):
        if tagged and ner_tag == ner_tag_id and (ner_tags[i-1] != ner_tag_id if i > 0 else True):
            string += begin_tag
        string += word
        if tagged and ner_tag == ner_tag_id and (ner_tags[i+1] != ner_tag_id if i < len(ner_tags)-1 else True):
            string += end_tag
        string += ' '
    return string.strip()
    

def sentences_with_most_occurences(dataset, example_index, ner_tag_id, n):
    counts = [e['ner_tags'].count(ner_tag_id) for e in dataset['train']]
    return sorted(range(len(counts)), key=lambda i: counts[i])[-n:]

def sentences_with_most_common_words(dataset, example_index, ner_tag_id, n):
    ref_words = dataset['test'][example_index]['words']
    counts = [len(set(e['words']).intersection(set(ref_words))) for e in dataset['train']]
    res = sorted(range(len(counts)), key=lambda i: counts[i])[-n:]
    return res

def sentences_with_closest_tf_idf(dataset, example_index, ner_tag_id, n):
    tokenized_examples = [e['words' if 'words' in e else 'tokens'] for e in dataset['train']]
    tokenized_examples.append(dataset['test'][example_index]['words' if 'words' in dataset['test'][example_index] else 'tokens'])
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf.fit(tokenized_examples)
    tfidf_matrix = tfidf.transform(tokenized_examples)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    res = sorted(range(len(similarities[0])), key=lambda i: similarities[0][i])[-n:]
    return res

criteria = {
    'most_occurences' : sentences_with_most_occurences,
    'most_common_words' : sentences_with_most_common_words,
    'closest_tf_idf' : sentences_with_closest_tf_idf
}

def make_prompt(dataset, example_index, ner_tag, ner_tag_id, language, domain, begin_tag, end_tag, n_few_shots, criterion):
    #this function takes an example and a ner tag and returns a prompt in english
    keywords = prompt_keywords[language]
    prompt = keywords['first_sentence'].format(keywords['domains_jobs'][domain], keywords['ner_tags'][ner_tag])
    #get the first example
    # few_shots = sentences_with_closest_tf_idf(dataset, example_index, ner_tag_id, 3)
    few_shots= criteria[criterion](dataset, example_index, ner_tag_id, n_few_shots)
    random.shuffle(few_shots)
    for i in few_shots:
        prompt+= keywords['input_intro']+example2string(dataset['train'][i], ner_tag_id, begin_tag, end_tag, tagged=False)+'\n'
        prompt+= keywords['output_intro']+example2string(dataset['train'][i], ner_tag_id, begin_tag, end_tag, tagged=True)+'\n'
    prompt+= keywords['last_sentence'].format(keywords['ner_tags'][ner_tag], begin_tag, end_tag)
    prompt+= keywords['input_intro']+example2string(dataset['test'][example_index], ner_tag_id, begin_tag, end_tag, tagged=False)+'\n'
    prompt+= keywords['output_intro']
    return prompt


prompt_keywords = {
    'en' : {
            'first_sentence' : "I am an expert {}, I can identify mentions of {} in a sentence. I can also format them. Here are some examples of sentences I can handle:\n",
            'last_sentence' : "Imitate me. Identify the mentions of {} in the following sentence, by putting \"{}\" in front and a \"{}\" behind the mention in the following sentence.\n",
            'domains_jobs' : {
                'clinical' : "clinician",
                'general' : "linguist"
            },
            'ner_tags' : {
                'PER' : "person names",
                'DISO' : "disorders",
                'LOC' : "places"
            },
            'input_intro' : "Input: ",
            'output_intro' : "Output: ",
        }
    ,
    'fr' : {
        'first_sentence' : "Je suis un {} expert, je sais identifier les mentions des {} dans une phrase. Je peux aussi les mettre en forme. Voici quelques exemples de phrases que je peux traiter :\n",
        #'last_sentence' : "Imite-moi. Identifie les mentions de {} dans la phrase suivante, en mettant \"{}\" devant et un \"{}\" derrière la mention dans la phrase suivante.\n",
        'last_sentence':"",
        'domains_jobs' : {
            'clinical' : "clinicien",
            'general' : "linguiste"
        },
        'ner_tags' : {
            'PER' : "noms de personnes",
            'DISO' : "maladies et symptômes",
            'LOC' : "lieux"
        },
        'input_intro' : "Entrée : ",
        'output_intro' : "Sortie : ",
    }
}


args = argparse.ArgumentParser()
args.add_argument("--language", type=str, default="fr", help="language of the dataset")
args.add_argument("--domain", type=str, default="general", help="domain of the dataset")
args.add_argument("--ner_tag", type=str, help="ner tag to evaluate")
args.add_argument("--begin_tag", type=str, default="@@")
args.add_argument("--end_tag", type=str, default="##")
args.add_argument("--n_few_shot", type=int, default=5)
args.add_argument("--model_name", type=str, default="bigscience/bloom-7b1")
args.add_argument("--batch_size", type=int, default=2)
args.add_argument("--criterion", type=str, default="closest_tf_idf")
args.add_argument("--overwrite_prompt_cache", action="store_true")
args = args.parse_args()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bloom_ner")

if args.domain == 'general':
    dataset_name = 'Jean-Baptiste/wikiner_fr'
    dataset = datasets.load_dataset(dataset_name)
    dataset['train'] = [example for example in dataset['train'] if len(example['tokens']) < 40]
    tag_to_id = {"O":0,"LOC":1,"PER":2,"FAC":3,"ORG":4}
    ner_tag = args.ner_tag if args.ner_tag else 'PER'
else :
    dataset_name = 'meczifho/QuaeroFrenchMed'
    dataset = datasets.load_dataset(dataset_name,'MEDLINE')
    tag_to_id = {"O":0,"ANAT":1,"LIVB":2,"DISO":3,"PROC":4,"CHEM":5,"GEOG":6,"PHYS":7,"PHEN":8,"OBJC":9,"DEVI":10}    
    ner_tag = args.ner_tag if args.ner_tag else 'DISO'

time_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logfile = open('log_'+time_date+'.txt', 'w')
logfile.write('language: '+args.language+'\n')
logfile.write('domain: '+args.domain+'\n')
logfile.write('ner_tag: '+ner_tag+'\n')
logfile.write('begin_tag: '+args.begin_tag+'\n')
logfile.write('end_tag: '+args.end_tag+'\n')
logfile.write('n_few_shot: '+str(args.n_few_shot)+'\n')
logfile.write('model_name: '+args.model_name+'\n')
logfile.write('criterion: '+args.criterion+'\n')
logfile.write('='*50+'\n')

tp_sum = 0
relevant_sum = 0
retrieved_sum = 0

assert args.criterion in criteria.keys(), "criterion must be in "+str(criteria.keys())
params = dataset_name+args.language+args.domain+ner_tag+args.begin_tag+args.end_tag+str(args.n_few_shot)+args.criterion
hash_object = hashlib.md5(params.encode())
if os.path.exists('prompts_'+hash_object.hexdigest()+'.txt') and not args.overwrite_prompt_cache:
    logger.info("Loading prompts...")
    with open('prompts_'+hash_object.hexdigest()+'.txt', 'r') as f:
        prompts = f.read().split('='*50)
    logger.info("Loaded prompts.")
else:
    logger.info("Making prompts...")
    prompts = []
    for i in tqdm(range(len(dataset['test']))):
        new_prompt = make_prompt(
            dataset, 
            i, 
            ner_tag, 
            tag_to_id[ner_tag], 
            args.language, 
            args.domain, 
            args.begin_tag, 
            args.end_tag, 
            args.n_few_shot,
            args.criterion,
        )
        prompts.append(new_prompt)

    #cache prompts
    with open('prompts_'+hash_object.hexdigest()+'.txt', 'w') as f:
        for prompt in prompts:
            f.write(prompt+'='*50)

logger.info("Generating outputs...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(args.model_name).to("cuda")
outputs = []
for i in tqdm(range(0, len(prompts), args.batch_size)):
    batch = prompts[i:i+args.batch_size]
    input_ids = tokenizer(batch, padding=True, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=40, do_sample=True, top_p=0.9, top_k=10, temperature=0.7)
    outputs.extend([tokenizer.decode(o, skip_special_tokens=True) for o in output])

logger.info("Evaluating...")
targets = [example2string(dataset['test'][i], tag_to_id[ner_tag], args.begin_tag, args.end_tag, tagged=True) for i in range(len(dataset['test']))]
for target, o in tqdm(zip(targets, outputs)):
    prediction = o.split('\n')[-1]
    #print target and predictions to a new log file
    logfile.write(target+'\n')
    logfile.write(prediction+'\n')
    logfile.write('-'*50+'\n')
    
    regex_begin_tag = re.escape(args.begin_tag)
    regex_end_tag = re.escape(args.end_tag)
    target_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', target)
    prediction_mentions = re.findall(r'(?<='+regex_begin_tag+').*?(?='+regex_end_tag+')', prediction)
    
    tp_sum += len(set(target_mentions).intersection(set(prediction_mentions)))
    relevant_sum += len(target_mentions)
    retrieved_sum += len(prediction_mentions)

print("precision: ", tp_sum/retrieved_sum if retrieved_sum > 0 else 0)
print("recall: ", tp_sum/relevant_sum if relevant_sum > 0 else 0)
print("f1: ", 2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0)
print("=====================================")

logfile.write("precision: "+str(tp_sum/retrieved_sum if retrieved_sum > 0 else 0)+'\n')
logfile.write("recall: "+str(tp_sum/relevant_sum if relevant_sum > 0 else 0)+'\n')
logfile.write("f1: "+str(2*tp_sum/(relevant_sum+retrieved_sum) if relevant_sum+retrieved_sum > 0 else 0)+'\n')
logfile.write("="*50+'\n')
logfile.close()