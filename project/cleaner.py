import langid
import pandas as pd

df = pd.read_csv('FILE', delimiter=",")

count = 0

for index, row in df.iterrows():
	text = row['Text']
	language = langid.classify(str(text))

	# if a different language is detected, drop the whole row
	if language[0] != 'en':
		print(text)
		print(language)
		df.drop(index, inplace=True)
		count += 1

df.to_csv("NEW FILE")
print("deleted", count)