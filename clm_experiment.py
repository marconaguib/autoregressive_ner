import datetime
import os
import argparse
import logging
import random
import json
from vllm import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from clm_predict import predict_for_dataset, MODEL_INSTRUCTION_TEMPLATES
from nlstruct import BRATDataset
from nlstruct.metrics import MetricsCollection
from nlstruct_extensions import HuggingfaceNERDataset, DocumentEntityMetricPerLabel
from nlstruct.data_utils import sentencize
from dataset_info import get_dataset_colnames, get_dataset_ner_tags, get_dataset_tag_map, get_dataset_language, get_dataset_specialist_name
from pred_utils import full_preds_string, get_metrics_string

args = argparse.ArgumentParser()
#MAIN ARGS
args.add_argument("--dataset_name", type=str, help="dataset name", default="conll2003")
args.add_argument('-d', "--load_dataset_from_disk", action="store_true", help="load dataset from disk, helpful on Jean Zay")
args.add_argument("--model_name", type=str, default="gpt2", help="model name")

#EXPERIMENT ARGS
args.add_argument('--no_write_log', dest='write_log', action='store_false')
args.add_argument('-n', '--n_gpus', type=int, default=1)
args.add_argument('--transformers', action="store_true")
args.add_argument('--debug', action="store_true")
args.add_argument('--log_full_preds', action="store_true")

#ABLATION ARGS
args.add_argument('--control', action="store_true")
args.add_argument('--random_seed', type=int, default=42)
args.add_argument('-p','--partition_seed', type=int, default=1)
args.add_argument('-s', '--training_size', type=int, default=100)
args.add_argument('--listing', action="store_true")
args.add_argument('--grid_search', action="store_true")

args = args.parse_args()
random.seed(args.random_seed)
time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
script_dir = os.path.dirname(__file__)
    
################# DATASET LOADING #################
try :
    doc_id_colname, words_colname, ner_tags_colname = get_dataset_colnames(args.dataset_name)
    dataset = HuggingfaceNERDataset(
        dataset_name=args.dataset_name,
        tag_map=get_dataset_tag_map(args.dataset_name),
        doc_id_colname=doc_id_colname,
        words_colname=words_colname,
        ner_tags_colname=ner_tags_colname,
        load_from_disk=args.load_dataset_from_disk,
    )
    #This is not supposed to be here, but WikiNER is a mess for now and I have no time to fix it
    if "WikiNER" in args.dataset_name:
        dataset.train_data = [e for e in dataset.train_data if e['doc_id'].startswith(args.dataset_name[-2:] if args.dataset_name[-1] != '/' else args.dataset_name[-3:-1])]
        dataset.test_data = [e for e in dataset.test_data if e['doc_id'].startswith(args.dataset_name[-2:] if args.dataset_name[-1] != '/' else args.dataset_name[-3:-1])]
except:
    dataset = BRATDataset(
        train= f"{args.dataset_name}/train",
        val= 0, 
        test= f"{args.dataset_name}/test",
    )

traindev_dataset = []
for e in dataset.train_data:
    sentences = sentencize(e, reg_split=r"(?<=[.|\s])(?:\s+)(?=[A-Z])", entity_overlap="split")
    traindev_dataset.extend([s for s in sentences if len(s['text']) < 512])
test_dataset = []
for e in dataset.test_data:
    sentences = sentencize(e, reg_split=r"(?<=[.|\s])(?:\s+)(?=[A-Z])", entity_overlap="split")
    test_dataset.extend([s for s in sentences if len(s['text']) < 512])
traindev_dataset_this_seed = random.Random(args.partition_seed).sample(traindev_dataset, args.training_size)
last_two_dirs = '-'.join(args.dataset_name.split('/')[-2:])
ner_tags = get_dataset_ner_tags(args.dataset_name)
dataset_language = get_dataset_language(args.dataset_name)
prompt_specialist_name = get_dataset_specialist_name(args.dataset_name, dataset_language)

if args.debug:
    traindev_dataset_this_seed = traindev_dataset_this_seed[:50]
    test_dataset = test_dataset[:50]
    args.training_size = 50

metrics = MetricsCollection({
    "exact": DocumentEntityMetricPerLabel(binarize_tag_threshold=1., binarize_label_threshold=1., add_label_specific_metrics=ner_tags, filter_entities=ner_tags),
    "partial": DocumentEntityMetricPerLabel(binarize_tag_threshold=1e-5, binarize_label_threshold=1., add_label_specific_metrics=ner_tags, filter_entities=ner_tags),
})

################# MODEL LOADING #################
if not args.transformers:
    compute_capability = torch.cuda.get_device_capability()
    llm = LLM(args.model_name, tensor_parallel_size=args.n_gpus, seed=args.random_seed, dtype="float16" if compute_capability[0]<8 else "auto", trust_remote_code=True)
    tokenizer = None
    model = None
else:
    llm=None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto",torch_dtype=torch.bfloat16)
    model = model.eval()

model_base_name = os.path.basename(args.model_name)

################# EXPERIMENT DEFINITION #################
def run_with_hyper_params(
        test_on_test_set=False,

        prompt_language="en",
        n_few_shot=5,
        one_step=True,

        taggers=("@@ ##",", "),
        prompt_youre_a_specialist=False,
        prompt_label_description=False,

        prompt_ask=False,
        prompt_long_answer=False,
        prompt_dash=False,
        ):
    logger.info(f"Running with hyperparams: {locals()}")
    #This is a function that will be called by the hyperparameter search
    folder_name = 'results'
    os.makedirs(os.path.join(script_dir, folder_name), exist_ok=True)
    res_dict = {}
    assert len(taggers) == 2, "taggers must be a tuple of two strings"
    assert len(taggers[0].split(' ')) == 2, "taggers must be a string with two words separated by a space"
    begin_tag, end_tag = taggers[0].split(' ')
    list_separator = taggers[1]

    res_dict['dataset_name'] = last_two_dirs
    res_dict['begin_tag'] = begin_tag
    res_dict['end_tag'] = end_tag
    res_dict['model_name'] = args.model_name
    res_dict['training_size'] = args.training_size
    res_dict['listing'] = args.listing
    res_dict['list_separator'] = list_separator
    res_dict['partition_seed'] = args.partition_seed
    res_dict['random_seed'] = args.random_seed
    res_dict['control'] = args.control
    res_dict['chat_template'] = MODEL_INSTRUCTION_TEMPLATES[args.model_name] if args.model_name in MODEL_INSTRUCTION_TEMPLATES else ""
    res_dict['ner_tags'] = ner_tags
    res_dict['first_example'] = traindev_dataset_this_seed[0]['text']
    res_dict['last_example'] = traindev_dataset_this_seed[-1]['text']
    res_dict['test_on_test_set'] = test_on_test_set
    res_dict['n_few_shot'] = n_few_shot
    res_dict['prompt_language'] = prompt_language
    res_dict['prompt_youre_a_specialist'] = prompt_youre_a_specialist
    res_dict['prompt_label_description'] = prompt_label_description
    res_dict['prompt_ask'] = prompt_ask
    res_dict['prompt_long_answer'] = prompt_long_answer
    res_dict['prompt_dash'] = prompt_dash
    res_dict['one_step'] = one_step

    model_kwargs = {
        "num_beams": 3,
        "do_sample": False,
        # "temperature": 0.,
        # "top_p": 0.,
    }
    res_dict.update(model_kwargs)

    logger.info("Generating...")
    textual_outputs, predicted_dataset, first_prompt_example, second_prompt_example = predict_for_dataset(
        llm=llm,
        model=model,
        tokenizer=tokenizer,
        training_data=traindev_dataset_this_seed,
        testing_data=test_dataset if test_on_test_set else None,
        ner_tags=ner_tags,
        model_name=args.model_name,
        control=args.control,
        model_kwargs=model_kwargs,
        random_seed=args.random_seed,
        prompt_specialist_name=prompt_specialist_name,
        listing=args.listing,
        list_separator=list_separator,
        
        #hyperparams
        n_few_shot=n_few_shot,
        one_step=one_step,
        prompt_language=prompt_language,
        prompt_youre_a_specialist=prompt_youre_a_specialist,
        prompt_label_description=prompt_label_description,
        prompt_ask=prompt_ask,
        prompt_long_answer=prompt_long_answer,
        prompt_dash=prompt_dash,
        begin_tag=begin_tag,
        end_tag=end_tag,
    )
    res_dict['first_prompt_example'] = first_prompt_example
    res_dict['second_prompt_example'] = second_prompt_example

    logger.info("Evaluating...")
    metric_dict = metrics(predicted_dataset, test_dataset if test_on_test_set else traindev_dataset_this_seed)
    for metric_name, metric_values in metric_dict.items():
        for k,v in metric_values.items():
            if not isinstance(v, int) and not isinstance(v, float):
                metric_dict[metric_name][k] = v.item()
            metric_dict[metric_name][k] = round(metric_dict[metric_name][k], 3)
    res_dict.update(metric_dict)
    logger.info(get_metrics_string(metric_dict, ner_tags))
    assert logfilename is not None #normally it should be defined
    if args.write_log:
        with open(logfilename, 'a') as logfile:
            logfile.write(get_metrics_string(metric_dict, ner_tags))
    
    if args.log_full_preds:
        full_preds = full_preds_string(textual_outputs, predicted_dataset, test_dataset if test_on_test_set else traindev_dataset_this_seed, ner_tags)
        full_preds_path = os.path.join(script_dir, folder_name)+f'/full_preds_{last_two_dirs}_{model_base_name}_{args.random_seed}_{time_str}.txt'
        res_dict['full_preds_path'] = full_preds_path
        with open(full_preds_path, 'w') as f:
            f.write(full_preds)
    if args.write_log:
        res_dict_path = os.path.join(script_dir, folder_name)+f'/res_dict_{last_two_dirs}_{model_base_name}_{args.random_seed}_{time_str}.json'
        with open(res_dict_path, 'w') as f:
            json.dump(res_dict, f)
    return metric_dict['exact']['f1']

################# HYPERPARAMETER SEARCH #################
possible_features = {
    "prompt_language": dataset_language,
    "n_few_shot": 10,
    "one_step": False,
    
    "taggers": ("<< >>", "\n"),
    "prompt_youre_a_specialist": True,
    "prompt_label_description": True,

    "prompt_ask": True,
    "prompt_long_answer": True,
    "prompt_dash": True,
}
log_dir = os.path.join(script_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
logfilename = os.path.join(log_dir, f"{last_two_dirs}_{model_base_name}_{args.random_seed}_{time_str}.log")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("experiment")

if not args.grid_search:
    #run once without any features
    logger.info("Running without any features")
    with open(logfilename, 'w') as logfile:
        logfile.write("Running without any features\n")
    best_f1 = run_with_hyper_params()
    kept_features = {}
    for feature_name, feature_value in possible_features.items():
        if feature_name == "prompt_language" and dataset_language == "en":
            #we don't want to test prompt_language if the dataset is already in english
            continue
        new_features = {feature_name: feature_value}
        #exceptionally, if the new feature is prompt_long_answer, we want to test it with one_step=False
        if feature_name == "prompt_long_answer" and "one_step" not in kept_features:
            new_features["one_step"] = False
        
        for k,v in new_features.items():
            logger.info(f"Testing feature {k} with value {v}")
            with open(logfilename, 'a') as logfile:
                logfile.write(f"Testing feature {k} with value {v}\n")

        #run with the new feature
        new_f1 = run_with_hyper_params(**kept_features, **new_features)

        #if the new feature is better, keep it
        if new_f1 > best_f1:
            for k,v in new_features.items():
                #this loop runs almost always only once, except for prompt_long_answer, where if we keep it, we also want to keep one_step=False
                logger.info(f"Feature {k} with value {v} kept")
                with open(logfilename, 'a') as logfile:
                    logfile.write(f"Feature {k} with value {v} kept\n")
                kept_features[k] = v
            best_f1 = new_f1
        else:
            for k,v in new_features.items():
                #this loop runs almost always only once, except for prompt_long_answer, where if we discard it, we also want to discard one_step=False
                logger.info(f"Feature {k} with value {v} discarded")
                with open(logfilename, 'a') as logfile:
                    logfile.write(f"Feature {k} with value {v} discarded\n")

    logger.info(f"Best F1: {best_f1}")
    logger.info(f"Best features: {kept_features}")
    with open(logfilename, 'a') as logfile:
        logfile.write(f"Best F1: {best_f1}\n")
        logfile.write(f"Best features: {kept_features}\n")
else:
    import itertools
    #make every possible combination of features
    all_features = []
    for i in range(len(possible_features)+1):
        all_features.extend(itertools.combinations(possible_features.items(), i))
    logger.info(f"Testing {len(all_features)} combinations of features")
    with open(logfilename, 'w') as logfile:
        logfile.write(f"Testing {len(all_features)} combinations of features\n")
    best_f1 = 0
    for features in all_features:
        logger.info(f"Testing features {features}")
        with open(logfilename, 'a') as logfile:
            logfile.write(f"Testing features {features}\n")
        new_features = dict(features)
        #exceptionally, if the new feature is prompt_long_answer, we want to test it with one_step=False
        if "prompt_long_answer" in new_features and "one_step" not in new_features:
            new_features["one_step"] = False
        
        #run with the new features
        new_f1 = run_with_hyper_params(**new_features)

        #if the new features are better, keep them
        if new_f1 > best_f1:
            logger.info(f"Features {features} kept")
            with open(logfilename, 'a') as logfile:
                logfile.write(f"Features {features} kept\n")
            kept_features = new_features
            best_f1 = new_f1
        else:
            logger.info(f"Features {features} discarded")
            with open(logfilename, 'a') as logfile:
                logfile.write(f"Features {features} discarded\n")  
    logger.info(f"Best F1: {best_f1}")
    logger.info(f"Best features: {kept_features}")
    with open(logfilename, 'a') as logfile:
        logfile.write(f"Best F1: {best_f1}\n")
        logfile.write(f"Best features: {kept_features}\n")      

#run with the best features on the test set
logger.info("Running with the best features on the test set")
with open(logfilename, 'a') as logfile:
    logfile.write("Running with the best features on the test set\n")
run_with_hyper_params(test_on_test_set=True, **kept_features)



    

