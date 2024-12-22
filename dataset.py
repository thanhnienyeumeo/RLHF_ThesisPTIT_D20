from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, Features
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import torch
from tqdm import tqdm
import random
import json
from tokenizer import TiktokenTokenizer


class DahoasSFTStaticPromptsDataset(Dataset):

    def __init__(self,
                 block_size,
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split="train")
        self.prompts = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading DahoasSFTStaticPromptsDataset")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']
            tokens = tokenizer(prompt,
                               max_length=block_size,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")

            self.prompts.append(
                [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])

            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("fka/awesome-chatgpt-prompts", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["prompt"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0], self.prompts[idx][1], self.prompts[idx][2]  # (1, T), (1, T)


class EYLSFTStaticDataset(Dataset):

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        if split == "train":
            with open("./sft_train.json") as fp:
                dataset = json.load(fp)
        else:
            with open("./sft_test.json") as fp:
                dataset = json.load(fp)
        self.tokens = []
        self.block_size = block_size
        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading EYLSFTStaticDataset {split} split")
        for chosen in dataset:
            cnt += 1
            response_text = chosen + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens from {cnt} examples.")

    def __len__(self):
        import sys
        return sys.maxsize

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        return x, y


class DahoasSFTStaticDataset(IterableDataset):
    """
    https://huggingface.co/datasets/Dahoas/sft-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset(
            "Dahoas/sft-static",
            revision="90e35d9cd625075f1224c4241734716ec9f0db78",
            split=split)
        self.tokens = []
        self.block_size = block_size

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading DahoasSFTStaticDataset {split} split")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']

            response_text += prompt + data['response'] + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

    def __iter__(self):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        yield x, y


class DahoasRMStaticDataset(Dataset):
    """
    https://huggingface.co/datasets/Dahoas/rm-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split=split)
        self.pairs = []
        self.masks = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading DahoasRMStaticDataset {split} split")
        for data in dataset:
            cnt += 1
            prompt = data['prompt']

            positive_text = prompt + data['chosen'] + "<|endoftext|>"
            positive = tokenizer(positive_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            negative_text = prompt + data['rejected'] + "<|endoftext|>"
            negative = tokenizer(negative_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            self.pairs.append(
                torch.stack((positive['input_ids'], negative['input_ids']),
                            dim=0))

            self.masks.append(
                torch.stack(
                    (positive['attention_mask'], negative['attention_mask']),
                    dim=0))
            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("Dahoas/rm-static", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["prompt"] + data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)


class AnthropicHHRLHFDataset(Dataset):
    """
    https://huggingface.co/datasets/Anthropic/hh-rlhf#dataset-summary
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        # if split == 'train':

        self.pairs = []
        self.masks = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        num_skip = 0
        for data in dataset:
            positive = tokenizer(data["chosen"],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=False,
                                 return_tensors="pt")
            positive_indices = positive["input_ids"]
            if len(positive_indices) > block_size:
                print("longer than block size")
                num_skip += 1
                continue
            positive_mask = positive["attention_mask"]

            negative = tokenizer(data["rejected"],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=False,
                                 return_tensors="pt")
            negative_indices = negative["input_ids"]
            negative_mask = negative["attention_mask"]
            if len(negative_indices) > block_size:
                print("longer than block size")
                num_skip += 1
                continue
            self.pairs.append(
                torch.stack((positive_indices, negative_indices), dim=0))

            self.masks.append(
                torch.stack((positive_mask, negative_mask), dim=0))
            cnt += 1
            if max_examples and cnt >= max_examples:
                break
        print(f'skip {num_skip} samples' )

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)

class ShareGPTDataset(Dataset):
    """
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        with open ("./ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json") as fp:
            dataset = json.load(fp)
        #shuffle the dataset with random seed
        random.seed(42)
        self.split = split
        random.shuffle(dataset)
        #split 10% for test
        if split == "train":
            dataset = dataset[:int(0.9*len(dataset))]
        else:
            dataset = dataset[int(0.9*len(dataset)):]
        self.tokens = []
        self.start_idx = []
        self.block_size = block_size
        tokenizer = TiktokenTokenizer('gpt2')
        cnt = 0
        print(f"Loading ShareGPTDataset")
        pre = 0
        for chosen in dataset:
            cnt += 1
            '''our dataset is in the form
            {
    "id": "QWJhYvA_0",
    "conversations": [
      {
        "from": "human",
        "value": "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients..."
      },
      {
        "from": "gpt",
        "value": "Here are the main ideas of Jeff Walker's Product Launch Formula that can be applied by a growth marketing agency for their clients:\n\n1. Identify the target audience and their needs: Understand the ideal customer for the product or service, and create a messaging that resonates with them.\n2. Pre-launch: Build anticipation and excitement for the launch by creating buzz, gathering testimonials and case studies, and using social media to create awareness.\n3. Launch: Use a well-crafted launch sequence to maximize sales and conversions. This can include offering bonuses, creating scarcity, and using a deadline to create urgency.\n4. Post-launch: Follow up with customers, gather feedback, and continue to provide value to keep them engaged and loyal.\n5. Create a product suite: Once the initial product is successful, expand the offering by creating additional products or services that address other needs of the same audience.\n6. Continual optimization: Continually monitor and optimize the launch process and product suite to improve results.\n7. Build a community: Use the launch process to build a community of customers who are passionate about the product and can help spread the word.\n8. Use automation: Use technology and automation to streamline the launch process and improve efficiency."
      },
      {
        "from": "human",
        "value": "Summarize the main ideas of Brendon Burchard's Experts Academy into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients..."
      },
'''
            chosen = chosen['conversations']
            #create conversation from the list
            response_text = ""
            for turn in chosen:
                    response_text += "\n\n"
                    if turn['from'] == "human":
                            
                        response_text += 'Human: ' + turn['value'] 
                    else:
                        response_text += 'Assistant:'  + turn['value']
            response_text += "<|endoftext|>"
                    # break
                    #tokenizer pảt
            response = tokenizer(response_text)
            self.tokens += response['input_ids']
            self.start_idx.append(pre)
            pre = len(self.tokens)
            if max_examples and cnt >= max_examples:
                break
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        
        print(f"Loaded {len(self.tokens)} tokens from {cnt} examples.")

    

    def __len__(self):
        if self.split == 'test':
            return len(self.start_idx)
        import sys
        return sys.maxsize

    def __getitem__(self, idx):
        if self.split == 'train':
            start = random.randint(0, len(self.tokens) - self.block_size - 2)
            x = self.tokens[start:start + self.block_size]
            y = self.tokens[start + 1:start + self.block_size + 1]
            return x, y
        else:
            if idx == 0:
                start = 0
            else:
                start = self.start_idx[idx]



class ShareGPTPromptsDataset(Dataset):

    def __init__(self,
                 block_size,
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        with open('../prompts.json') as fp:
            dataset = json.load(fp)
        self.prompts = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading ShareGPTPromptsDataset")
        for data in dataset:
            cnt += 1
            prompt = data
            tokens = tokenizer(prompt,
                               max_length=block_size,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")

            self.prompts.append(
                [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])

            if max_examples and cnt >= max_examples:
                break


    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0], self.prompts[idx][1], self.prompts[idx][2]  # (1, T), (1, T)
    
class SafeRLHFPromptsDataset(Dataset):

    def __init__(self,
                 block_size,
                    split = 'train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        

        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", "default", split = split)
        dataset.shuffle()
        self.prompts = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading SafeRLHFPromptsDataset")
        self.raw_prompts = []
        for data in dataset:
            # print(data)
            cnt += 1
            prompt = 'Human: ' + data['prompt'] +'\n\nAssistant: '
            # print(prompt)
            tokens = tokenizer(prompt,
                               max_length=block_size,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")

            self.prompts.append(
                [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])
            self.raw_prompts.append(prompt)
            if max_examples and cnt >= max_examples:
                break


    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0], self.prompts[idx][1], self.prompts[idx][2]  # (1, T), (1, T)
    

class MathDataset(Dataset):

    """
    https://huggingface.co/datasets/Dahoas/sft-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset1 = load_dataset(
            "openai/gsm8k",
            'main',
            split=split)
        dataset2 = load_dataset("lighteval/MATH",  'all', split = split, trust_remote_code=True)
        self.tokens = []
        self.block_size = block_size

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading MathDataset {split} split")
        for data in dataset1:
            response_text = 'Human: '
            cnt += 1
            prompt = data['question']

            response_text += prompt + "\n\n" + 'Assistant: ' + data['answer'] + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break
        for data in dataset2:
            response_text = 'Human: '
            cnt += 1
            prompt = data['problem']

            response_text += prompt + "\n\n" + 'Assistant: ' + data['solution'] + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        try:
            print(len(self.tokens))
        except:
            print('error')
            pass

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        yield x, y
    def __iter__(self):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        yield x, y
    def __len__(self):
        import sys
        return sys.maxsize
    
class StepDPODataset(Dataset):
    """
    https://huggingface.co/datasets/step-dpo
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Colder203/RLHF_Math_Step_GPT2", split='train')
        split_rate = 85/100
        if split == 'train':
            dataset = dataset.select(range(int(len(dataset)*split_rate)))
        else:
            dataset = dataset.select(range(int(len(dataset)*split_rate), len(dataset)))
        self.pairs = []
        self.masks = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        num_skip = 0
        for data in dataset:
            d1 =data["full_chosen"] + "<|endoftext|>"
            d2 = data["full_rejected"] + "<|endoftext|>"
            if len(d1) > block_size:
                d1 = d1[-block_size:]
            if len(d2) > block_size:
                d2 = d2[-block_size:]
            if d1 == d2:
                print('skip....')
                continue
            positive = tokenizer(data["full_chosen"] + "<|endoftext|>",
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            positive_indices = positive["input_ids"]
            positive_mask = positive["attention_mask"]
            # if len(positive_indices) > block_size:
            #     print(f"{len(positive_indices)} longer than block size")
            #     num_skip += 1
            negative = tokenizer(data["rejected"] ,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            negative_indices = negative["input_ids"]
            negative_mask = negative["attention_mask"]

            self.pairs.append(
                torch.stack((positive_indices, negative_indices), dim=0))

            self.masks.append(
                torch.stack((positive_mask, negative_mask), dim=0))
            cnt +=1
            negative = tokenizer(data["full_rejected"] + "<|endoftext|>",
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            negative_indices = negative["input_ids"]
            negative_mask = negative["attention_mask"]

            self.pairs.append(
                torch.stack((positive_indices, negative_indices), dim=0))

            self.masks.append(
                torch.stack((positive_mask, negative_mask), dim=0))
            cnt += 1
            
            if max_examples and cnt >= max_examples:
                break

    @classmethod
    def save(cls, split, fp):
        dataset = load_dataset("step-dpo", split=split)
        examples = []
        for data in tqdm(dataset):
            examples.append(data["chosen"])
        import json
        json.dump(examples, fp)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)
    
class StepDPOPromptsDataset(Dataset):

    def __init__(self,
                 block_size,
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        path = 'Colder203/RLHF_PPO_Math_Step'
        dataset = load_dataset(path, split='train')
        self.prompts = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading StepDPOPromptsDataset")
        for data in dataset['prompt']:
            cnt += 1
            # prompt = data['prompt']
            prompt = data
            tokens = tokenizer(prompt,
                               max_length=block_size,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")

            self.prompts.append(
                [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])

            if max_examples and cnt >= max_examples:
                break
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0], self.prompts[idx][1], self.prompts[idx][2]  # (1, T), (1, T)


class MetaMathDataset(Dataset):

    """
    https://huggingface.co/datasets/Dahoas/sft-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset(
            "meta-math/MetaMathQA",
            
            split='train')
        split_rate  = 0.95
        if split == 'train':
            dataset = dataset.select(range(int(len(dataset)*split_rate)))
        else:
            dataset = dataset.select(range(int(len(dataset)*split_rate), len(dataset)))
        self.tokens = []
        self.block_size = block_size

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        print(f"Loading MetaMathDataset {split} split")
        for data in dataset:
            response_text = 'Human: '
            cnt += 1
            prompt = data['query']

            response_text += prompt + "\n\n" + 'Assistant: ' + data['response'] + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        try:
            print(len(self.tokens))
        except:
            print('error')
            pass

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        return x, y
    def __iter__(self):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        yield x, y
    def __len__(self):
        import sys
        return sys.maxsize