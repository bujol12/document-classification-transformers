from json_document_dataset import JsonDocumentDataset
from config import Config

conf = Config("roberta-base", True)
data = JsonDocumentDataset("../data/processed/fce/json/test.json", conf)

count = 0
for i in range(len(data)):
    if len(data[i][0]['input_ids']) > 512:
        print(len(data[i][0]['input_ids']))
        count += 1

print(
    f"Total documents: {len(data)}, documents with >512 BERT tokens: {count}, {round(count * 100 / len(data))} % of total")
