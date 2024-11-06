
# %%
import os
import pickle
import json
import copy
from tqdm import tqdm
from collections import defaultdict
from openai import OpenAI

client = OpenAI(api_key="sk-2EReUlr7NvBL1xIKApTxqesBx0gpDLQTmhbtbRsxIRAy2drF", base_url="https://api.ai.cs.ac.cn/v1")
#client = OpenAI(base_url="https://api.ai.cs.ac.cn/v1")
# %%
# Load necessary data
id_map = json.load(open("./processed/id_map.json"))
item_dict = json.load(open("./processed/item2attributes.json", "r"))

# %%
def load_dataset():
    '''Load train, validation, test dataset'''
    User = defaultdict(list)
    with open('./processed/inter.txt', 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u, i = int(u), int(i)
            User[u].append(i)

    user_train = {}
    for user, items in User.items():
        if len(items) < 3:
            user_train[user] = items
        else:
            user_train[user] = items[:-2]

    return user_train

# %%
inter = load_dataset()
prompt_template = "The user has purchased the following beauty items: \n<HISTORY> \nBased on this history, please provide a brief user profile describing the user's preferences and style in 2-3 sentences."
def get_user_prompt(history):
    user_str = copy.deepcopy(prompt_template)
    hist_str = ""
    for index, item in enumerate(history, start=1):
        try:
            item_str = item_dict[id_map["id2item"][str(item)]]["title"]
            hist_str = hist_str + f"{index}. {item_str}, "
        except:
            continue

    # limit the prompt length
    if len(hist_str) > 8000:
        hist_str = hist_str[-8000:]

    user_str = user_str.replace("<HISTORY>", hist_str)
    return user_str

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.5,  # Lower temperature for less random answers
        messages=[
            {"role": "system", "content": "You are an expert beauty product analyst and consumer behavior specialist. Your task is to create detailed user profiles based on their beauty product purchase history, identifying key trends and preferences in their evolving shopping behaviors. Note that the purchase history is in reversed chronological order, with smaller indices representing more recent purchases."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Based on the user's purchase history, please provide the following in JSON format:\n1. A concise 2-3 sentence user profile describing their beauty preferences, style, and potential skincare or makeup needs.\n2. A list of key purchased items that best represent the user's evolving beauty routine or preferences. Include the item's index in the shopping history, remembering that smaller indices indicate more recent purchases.\n3. A brief explanation of why these items are significant for understanding the user's beauty habits and predicting future purchases.\nUse this JSON structure:\n{\n    \"user_profile\": \"2-3 sentence description of user's beauty preferences and style\",\n    \"key_behaviors\": [\n        {\n            \"index\": \"Item's index in the shopping history (smaller index means more recent purchase)\",\n            \"item\": \"Name of the key purchased item\"\n        },\n        ...\n    ],\n    \"reason\": \"Explanation of why these items are significant for the user's beauty routine and future next item recommendations, considering the chronological order of purchases\"\n}"}
        ]
    )
    return response.choices[0].message.content.strip()

def parse_llm_response(response):
    try:
        json_response = json.loads(response)
        user_profile = json_response.get("user_profile", "")
        key_items = [item.get("index", "") for item in json_response.get("key_behaviors", [])]
        rationale = json_response.get("reason", "")
        return user_profile, key_items, rationale
    except json.JSONDecodeError:
        # If JSON parsing fails, return empty values
        return "", [], ""
# %%
# Generate user profiles
user_profiles = {}
key_items_dict = {}
rationale = {}
i = 0
for user, history in tqdm(inter.items()): 
    user_prompt = get_user_prompt(history)
    # user_profiles[user] = ask_llm(user_prompt)
    response = ask_llm(user_prompt)
    user_profiles[user], key_items_dict[user], rationale[user] = parse_llm_response(response)
    key_items_dict[user] = [int(item) for item in key_items_dict[user]]
    print(user_prompt)
    print(response)
    print(user_profiles[user])
    print(key_items_dict[user])
    print(rationale[user])
    i += 1
    if i > 100:
        break
# %%
# make a new folder called parsed
os.makedirs("./parsed", exist_ok=True)
# Save user profiles
json.dump(user_profiles, open("./parsed/user_profiles.json", "w"))
json.dump(key_items_dict, open("./parsed/key_items_dict.json", "w"))
json.dump(rationale, open("./parsed/rationale.json", "w"))

# %%
# assume also generated_rec_items.json : in shape of {user_id: generated_rec_item_text}
# note that the generated_rec_item_text is generated by asking LLM for giving the next item description for the user and to be conducted before the following step

# %%
# load generated_rec_items.json
generated_rec_items = json.load(open("./generated_rec_items.json", "r"))
# for each user_id, find the most similar item in the candidate pool (item_id in item_dict) to the generated_rec_item_text
# calculate the similarity using cosine similarity between the generated_rec_item_text and the item_description in item_dict
# save the similarity score in a new file called generated_rec_items_similarity.json    
# get the top-k similar items for each user_id
# save the top-k similar item ids for each user_id in a new file called generated_rec_items_topk.json

# %%
# generate rec_text for each item_id
   #attention_marker_path = './data/' + dataset_name + '/parsed/attention_marker.npy' 
    #llm_top_k_load_path = os.path.join('./data/', args.dataset, '/parsed/', 'llm_top_k_items.npy')