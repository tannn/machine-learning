import re
import pandas as pd

df = pd.read_csv('FILE', delimiter=",")
deleted = 0
count = 0

for index, row in df.iterrows():
	text = str(row['Text'])
	if text[0:2] == "b'" or text[0:2] == 'b"':
		text = text[2:-1]

	p = re.compile(r'\\n', re.IGNORECASE)
	m = re.compile(r'\\x..', re.IGNORECASE)
	l = re.compile(r'https:\/\/t\.co\/[A-Za-z0-9]{10}')

	replace_n = p.sub('', text)
	replace_n_r = m.sub('', replace_n)
	final = l.sub('', replace_n_r)
	print(final)

	if final == "nan" or not final:
		df.drop(index, inplace=True)
		deleted += 1
	else:
		df.at[index,'Text'] = final
		count += 1

df.to_csv("NEW FILE")
print("deleted",deleted)
print("changed",count)