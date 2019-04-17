import pandas as pd

df = pd.read_csv('labeled_first_half.csv', delimiter=",")

count = 0
# deleted = 0

for index, row in df.iterrows():
	label = row['Label']
	# identifier = row['ID']
	text = row['Text']

	# if not identifier or identifier == "nan":
	# 	df.drop(index, inplace=True)
	# 	deleted += 1
	# print()
	# if text == "nan" or not text:
	# 	df.drop(index, inplace=True)
	# 	deleted += 1
	if label != 0.0 or label != 1.0:
		df.at[index,'Label'] = 0
		count += 1

df.to_csv("relabeled_first_half.csv")
# print("deleted",deleted)
print("changed",count)