import json

datas = []
preds = []
with open('data/test/test.json', encoding='utf-8')as file:
    for line in file.readlines():
        data = json.loads(line.strip())
        datas.append(data)

with open('prediction.json', encoding='utf-8')as file:
    for line in file.readlines():
        data = json.loads(line.strip())
        preds.append(data)

for i, j in zip(datas, preds):
    i['sql'] = j

with open('result.json', encoding='utf-8', mode='w')as file:
    for d in datas:
        file.write(json.dumps(d, ensure_ascii=False) + '\n')
