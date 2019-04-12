import langid
import pandas as pd

df = pd.read_csv('second_half.csv', delimiter=",")

for index, row in df.iterrows():
	text = row['Text']
	language = langid.classify(str(text))

	# if a different language is detected, drop the whole row
	if language[0] != 'en':
		print(text)
		print(language)
		df.drop(index, inplace=True)

df.to_csv("cleaned_second_half.csv")
