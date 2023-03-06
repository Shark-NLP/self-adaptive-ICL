export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=wyx001  # change to your wandb account
export WANDB_API_KEY=78f39d1d23c9ded73f35de542ecb99e4a98ed6cb  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false

export HYDRA_FULL_ERROR=1

port=12715
#
num_ice=8
num_candidates=30

model_name=gpt2-xl
n_tokens=700
inf_batch_size=12
score_method="entropy"
force_topk=false
instruction_template=1
span=true
window=10
dataset_split="test"
rand_seed=1
root=/mnt/cache/wangyaoxiang/codes/adaptive
emb_field=X # or ALL
n_gpu=1

for task_name in sst2
do
  run_dir=${root}/output/${task_name}/${model_name}/${rand_seed}/${dataset_split}
  retrieve_file=${run_dir}/retrieved.json
  retrieve_file2=${run_dir}/retrieved2_entropy.json
  pred_file=${run_dir}/pred5.json
  mkdir -p ${run_dir}

  python prerank.py output_file=${retrieve_file} \
      emb_field=${emb_field} \
      num_ice=${num_ice} \
      num_candidates=${num_candidates} \
      dataset_reader.task_name=${task_name} \
      rand_seed=${rand_seed} \
      dataset_reader.dataset_split=${dataset_split} \
      index_reader.task_name=${task_name} \
      index_file=${run_dir}/index \
      scale_factor=0.1

  accelerate launch --num_processes ${n_gpu} --main_process_port ${port} retriever.py output_file=${retrieve_file2} \
      num_ice=${num_ice} \
      window=${window} \
      rand_seed=${rand_seed} \
      force_topk=${force_topk} \
      instruction_template=${instruction_template} \
      span=${span}\
      dataset_reader.task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      batch_size=${inf_batch_size} \
      method=${score_method}

  accelerate launch --num_processes ${n_gpu} --main_process_port ${port}  ppl_inferencer.py \
      dataset_reader.task_name=${task_name} \
      rand_seed=${rand_seed} \
      dataset_reader.dataset_path=${retrieve_file2} \
      instruction_template=${instruction_template} \
      span=${span} \
      dataset_reader.n_tokens=${n_tokens} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size}
done
