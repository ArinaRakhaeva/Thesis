import json
import requests 
import pandas as pd

dictionary_inflation = ['CPI', 'deflation', 'inflation','disinflation', 'inflationary', 'recession','stagflation','consumption basket','gas','oil','petrol','fuel','electricity','price','cost','income','revenue','wage','expenditure','payment','rent','purchasing power','tariff','sale']

def items_in_dictionary(dictionary, string):
    for term in dictionary:
        if term in string:
            return string
    return None

def download_json(url, destination):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded JSON file to {destination}")
    else:
        print(f"Failed to download JSON file. Status code: {response.status_code}")

month = "1"
year = "2022"

json_url = "https://api.nytimes.com/svc/archive/v1/" + year + "/" + month + ".json?api-key=9QigMRakLuhywBq0jduwbEZEyKV3XoiJ"
output_file = 'output.json'

download_json(json_url, output_file)

json_data = open('output.json')
data = json.load(json_data)
snippets_with_ids = []
lead_paragraph_with_ids = []
abstract_with_ids = []

for idx, doc in enumerate(data["response"]["docs"]):
    try:
        snippet = doc.get("snippet", "")
        cleaned_snippets = items_in_dictionary(dictionary_inflation, snippet)
        snippets_with_ids.append({"id": idx, "snippet": cleaned_snippets})
    except:
        continue
    try: 
        lead_paragraph = doc.get("lead_paragraph", "")
        cleaned_lead_paragraph = items_in_dictionary(dictionary_inflation, lead_paragraph)
        lead_paragraph_with_ids.append({"id": idx, "lead_paragraph": cleaned_lead_paragraph})
    except:
        continue
    try:
        abstract = doc.get("abstract", "")
        cleaned_abstract = items_in_dictionary(dictionary_inflation, abstract)
        abstract_with_ids.append({"id": idx, "abstract": cleaned_abstract})
    except:
        continue

for item in snippets_with_ids:
    print(f"ID: {item['id']}, Snippet: {item['snippet']}")

snippet_df = pd.DataFrame(snippets_with_ids, index = None)
#print(snippet_df)
lead_paragraph_df = pd.DataFrame(lead_paragraph_with_ids, index = None)
abstract_df = pd.DataFrame(abstract_with_ids, index = None)

merge_sn_lp = pd.concat([snippet_df,lead_paragraph_df,abstract_df], axis=1)
merge_sn_lp = merge_sn_lp.drop(merge_sn_lp.columns[0], axis=1)
merge_sn_lp = merge_sn_lp.dropna(subset=['snippet', 'lead_paragraph','abstract'], how='all')

print(merge_sn_lp)

#for item in lead_paragraphs_with_ids:
#    print(f"ID: {item['id']}, Lead_paragraph: {item['lead_paragraph']}")