import re
import pandas as pd

df = pd.read_csv('tweets.csv', delimiter=",")
deleted = 0
count = 0

for index, row in df.iterrows():
	text = str(row['tweets'])

	p = re.compile(r'ENGLISH TRANSLATION:?', re.IGNORECASE)
	m = re.compile(r'RT', re.IGNORECASE)
	l = re.compile(r'https?.*')
	z = re.compile(r'[^a-zA-Z0-9\\\/\|\*\.\?\!\@\#\$\%\^\&\*\(\)\_\-\+\=\:\;\"\'\{\}\[\]\, ]') #not quite working

	replace_n = p.sub('', text)
	replace_n_r = m.sub('', replace_n)
	replace_n_r_l = l.sub('', replace_n_r)
	final = z.sub('', replace_n_r_l)
	print(final)

	if final == "nan" or not final:
		df.drop(index, inplace=True)
		deleted += 1
	else:
		df.at[index,'tweets'] = final
		count += 1

	label = row['label']
	df.at[index,'label'] = 1

df.to_csv("modified-tweets.csv")
print("deleted",deleted)
print("changed",count)