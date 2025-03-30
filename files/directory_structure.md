# Neuron intervention for long-tail lifting

neuron = ["boost","supress"]
vector = ["mean","longtail"]
intervention = ["base", "zero", "random","mean","scaled","full"]
eval_set = ["merged","longtail_words"]
model = ["70m","410m"]

## Result

### Surprisal

Surprisal_ROOT = surprisal_dir
Surprisal_ROOT / neuron / vector / intervention / eval_set / model / pythia-{para_size}-deduped_{neuron_num}.csv



### identified neurons

Neuron_ROOT = neuron_dir
Neuron_ROOT / neuron / vector / intervention / model / pythia-{para_size}-deduped / 500_{neuron_num}.csv
