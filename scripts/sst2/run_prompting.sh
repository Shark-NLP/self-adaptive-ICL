export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=xx  # change to your wandb account
export WANDB_API_KEY=xx  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false

export HYDRA_FULL_ERROR=1

port=1277
num_ice=0
model_name=gpt2-xl
n_tokens=700
inf_batch_size=12

instruction_template=1
span=true
dataset_split="test"
rand_seed=1
root=/mnt/cache/wangyaoxiang/codes/adaptive
emb_field=X # or ALL
n_gpu=1

for task_name in sst2
do
  run_dir=${root}/output/${task_name}/${model_name}/${rand_seed}/${dataset_split}
  retrieve_file=${run_dir}/retrieved_0example.json
  pred_file=${run_dir}/pred6.json
  mkdir -p ${run_dir}

  python prerank.py output_file=${retrieve_file} \
      num_ice=${num_ice} \
      dataset_reader.task_name=${task_name} \
      rand_seed=${rand_seed} \
      dataset_reader.dataset_split=${dataset_split} \
      index_reader.task_name=${task_name} \
      index_file=${run_dir}/index \
      scale_factor=0.1

  accelerate launch --num_processes ${n_gpu} --main_process_port ${port}  ppl_inferencer.py \
      dataset_reader.task_name=${task_name} \
      rand_seed=${rand_seed} \
      dataset_reader.dataset_path=${retrieve_file} \
      instruction_template=${instruction_template} \
      span=${span} \
      dataset_reader.n_tokens=${n_tokens} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size}
done
