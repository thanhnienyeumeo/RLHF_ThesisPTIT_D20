from huggingface_hub import HfApi, HfFolder, Repository
# from credential import HUGGINGFACE_TOKEN
# Define your repository details
# repo_name = "Colder203/PPO_model_GPT2_Medium"
repo_name = 'Colder203/ppo_gpt2_medium_running'
checkpoint_path = "runs\ppo_fail"
# token = HUGGINGFACE_TOKEN

# Initialize the API and repository
api = HfApi()
# repo_url = api.create_repo(repo_name, token=token)
# repo = Repository(local_dir=checkpoint_path, clone_from=repo_url)

# upload_file
import os
list_dir = os.listdir(checkpoint_path)
for file in list_dir:
    api.upload_file(
        path_or_fileobj= os.path.join(checkpoint_path, file),
        path_in_repo='optimizer.pt',
        repo_id = 'Colder203/llama_finetune_metamath',
        repo_type='model'
    )