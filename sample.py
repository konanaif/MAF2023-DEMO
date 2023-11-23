import os
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
from glob import glob
from PIL import Image

from aif360.datasets import StandardDataset
from MAF2023.DataSet import aifData


compas_mappings = {
	'label_maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}],
	'protected_attribute_maps': [{0.0: 'Male', 1.0: 'Female'},
								{1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
}

def compas_preprocessing(df):
	"""Perform the same preprocessing as the original analysis:
	https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
	"""
	return df[(df.days_b_screening_arrest <= 30)
			& (df.days_b_screening_arrest >= -30)
			& (df.is_recid != -1)
			& (df.c_charge_degree != 'O')
			& (df.score_text != 'N/A')]
			

class CompasDataset(StandardDataset):
	def __init__(self, filepath='./Sample/compas-scores-two-years.csv',
		label_name='two_year_recid', favorable_classes=[0],
		protected_attribute_names=['sex', 'race'],
		privileged_classes=[['Female'], ['Caucasian']],
		instance_weights_name=None,
		categorical_features=['age_cat', 'c_charge_degree','c_charge_desc'],
		features_to_keep=['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count',
						'priors_count', 'c_charge_degree', 'c_charge_desc','two_year_recid'],
		features_to_drop=[], na_values=[],
		custom_preprocessing=compas_preprocessing,
		metadata=compas_mappings):

		try:
			df = pd.read_csv(filepath, index_col='id', na_values=na_values)
		except IOError as err:
			print("IOError: {}".format(err))
			print("To use this class, please download the following file:")
			print("\n\thttps://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
			print("\nand place it, as-is, in the folder:")
			print("\n\t{}\n".format(filepath))
			import sys
			sys.exit(1)

		super(CompasDataset, self).__init__(df=df, label_name=label_name,
			favorable_classes=favorable_classes,
			protected_attribute_names=protected_attribute_names,
			privileged_classes=privileged_classes,
			instance_weights_name=instance_weights_name,
			categorical_features=categorical_features,
			features_to_keep=features_to_keep,
			features_to_drop=features_to_drop, na_values=na_values,
			custom_preprocessing=custom_preprocessing, metadata=metadata)


german_mappings = {
	'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
	'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
								{1.0: 'Old', 0.0: 'Young'}],
}

def german_preprocessing(df):
	"""Adds a derived sex attribute based on personal_status."""
	# TODO: ignores the value of privileged_classes for 'sex'
	status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
				  'A92': 'female', 'A95': 'female'}
	df['sex'] = df['personal_status'].replace(status_map)

	return df

class GermanDataset(StandardDataset):
	"""German credit Dataset.
	See :file:`aif360/data/raw/german/README.md`.
	"""

	def __init__(self, filepath='./Sample/german.data',
		label_name='credit', favorable_classes=[1],
		protected_attribute_names=['sex', 'age'],
		privileged_classes=[['male'], lambda x: x > 25],
		instance_weights_name=None,
		categorical_features=['status', 'credit_history', 'purpose',
		'savings', 'employment', 'other_debtors', 'property',
		'installment_plans', 'housing', 'skill_level', 'telephone',
		'foreign_worker'],
		features_to_keep=[], features_to_drop=['personal_status'],
		na_values=[], custom_preprocessing=german_preprocessing,
		metadata=german_mappings):
		"""See :obj:`StandardDataset` for a description of the arguments.
		By default, this code converts the 'age' attribute to a binary value
		where privileged is `age > 25` and unprivileged is `age <= 25` as
		proposed by Kamiran and Calders [1]_.
		References:
			.. [1] F. Kamiran and T. Calders, "Classifying without
			   discriminating," 2nd International Conference on Computer,
			   Control and Communication, 2009.
		Examples:
			In some cases, it may be useful to keep track of a mapping from
			`float -> str` for protected attributes and/or labels. If our use
			case differs from the default, we can modify the mapping stored in
			`metadata`:
			>>> label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
			>>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
			>>> gd = GermanDataset(protected_attribute_names=['sex'],
			... privileged_classes=[['male']], metadata={'label_map': label_map,
			... 'protected_attribute_maps': protected_attribute_maps})
			Now this information will stay attached to the dataset and can be
			used for more descriptive visualizations.
		"""

		# as given by german.doc
		column_names = ['status', 'month', 'credit_history',
			'purpose', 'credit_amount', 'savings', 'employment',
			'investment_as_income_percentage', 'personal_status',
			'other_debtors', 'residence_since', 'property', 'age',
			'installment_plans', 'housing', 'number_of_credits',
			'skill_level', 'people_liable_for', 'telephone',
			'foreign_worker', 'credit']
		try:
			df = pd.read_csv(filepath, sep=' ', header=None, names=column_names,
							 na_values=na_values)
			df['credit'] = df['credit'] - 1
		except IOError as err:
			print("IOError: {}".format(err))
			print("To use this class, please download the following files:")
			print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
			print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc")
			print("\nand place them, as-is, in the folder:")
			print("\n\t{}\n".format(filepath))
			import sys
			sys.exit(1)

		super(GermanDataset, self).__init__(df=df, label_name=label_name,
			favorable_classes=favorable_classes,
			protected_attribute_names=protected_attribute_names,
			privileged_classes=privileged_classes,
			instance_weights_name=instance_weights_name,
			categorical_features=categorical_features,
			features_to_keep=features_to_keep,
			features_to_drop=features_to_drop, na_values=na_values,
			custom_preprocessing=custom_preprocessing, metadata=metadata)





adult_mappings = {
	'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
	'protected_attribute_maps': [{1.0: 'White', 0.0: 'Non-white'},
								 {1.0: 'Male', 0.0: 'Female'}]
}

class AdultDataset(StandardDataset):
	"""Adult Census Income Dataset.
	See :file:`aif360/data/raw/adult/README.md`.
	"""

	def __init__(self, file_directory='./Sample',
		label_name='income-per-year',
		favorable_classes=['>50K', '>50K.'],
		protected_attribute_names=['race', 'sex'],
		privileged_classes=[['White'], ['Male']],
		instance_weights_name=None,
		categorical_features=['workclass', 'education',
							'marital-status', 'occupation', 'relationship', 'native-country'],
		features_to_keep=[], features_to_drop=['fnlwgt'],
		na_values=['?'], custom_preprocessing=None,
		metadata=adult_mappings):
		"""See :obj:`StandardDataset` for a description of the arguments.
		Examples:
			The following will instantiate a dataset which uses the `fnlwgt`
			feature:
			>>> from aif360.datasets import AdultDataset
			>>> ad = AdultDataset(instance_weights_name='fnlwgt',
			... features_to_drop=[])
			WARNING:root:Missing Data: 3620 rows removed from dataset.
			>>> not np.all(ad.instance_weights == 1.)
			True
			To instantiate a dataset which utilizes only numerical features and
			a single protected attribute, run:
			>>> single_protected = ['sex']
			>>> single_privileged = [['Male']]
			>>> ad = AdultDataset(protected_attribute_names=single_protected,
			... privileged_classes=single_privileged,
			... categorical_features=[],
			... features_to_keep=['age', 'education-num'])
			>>> print(ad.feature_names)
			['education-num', 'age', 'sex']
			>>> print(ad.label_names)
			['income-per-year']
			Note: the `protected_attribute_names` and `label_name` are kept even
			if they are not explicitly given in `features_to_keep`.
			In some cases, it may be useful to keep track of a mapping from
			`float -> str` for protected attributes and/or labels. If our use
			case differs from the default, we can modify the mapping stored in
			`metadata`:
			>>> label_map = {1.0: '>50K', 0.0: '<=50K'}
			>>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
			>>> ad = AdultDataset(protected_attribute_names=['sex'],
			... categorical_features=['workclass', 'education', 'marital-status',
			... 'occupation', 'relationship', 'native-country', 'race'],
			... privileged_classes=[['Male']], metadata={'label_map': label_map,
			... 'protected_attribute_maps': protected_attribute_maps})
			Note that we are now adding `race` as a `categorical_features`.
			Now this information will stay attached to the dataset and can be
			used for more descriptive visualizations.
		"""

		train_path = os.path.join(file_directory, 'adult.data')
		test_path = os.path.join(file_directory, 'adult.test')
		# as given by adult.names
		column_names = ['age', 'workclass', 'fnlwgt', 'education',
			'education-num', 'marital-status', 'occupation', 'relationship',
			'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
			'native-country', 'income-per-year']
		try:
			train = pd.read_csv(train_path, header=None, names=column_names,
				skipinitialspace=True, na_values=na_values)
			test = pd.read_csv(test_path, header=0, names=column_names,
				skipinitialspace=True, na_values=na_values)
		except IOError as err:
			print("IOError: {}".format(err))
			print("To use this class, please download the following files:")
			print("\n\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
			print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test")
			print("\thttps://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names")
			print("\nand place them, as-is, in the folder:")
			print("\n\t{}\n".format(os.path.abspath(file_directory)))
			import sys
			sys.exit(1)

		df = pd.concat([test, train], ignore_index=True)

		super(AdultDataset, self).__init__(df=df, label_name=label_name,
			favorable_classes=favorable_classes,
			protected_attribute_names=protected_attribute_names,
			privileged_classes=privileged_classes,
			instance_weights_name=instance_weights_name,
			categorical_features=categorical_features,
			features_to_keep=features_to_keep,
			features_to_drop=features_to_drop, na_values=na_values,
			custom_preprocessing=custom_preprocessing, metadata=metadata)


class PubFigDataset(StandardDataset):
	ROOT = './Sample/pubfig'
	ATTRIBUTE_FILE = './Sample/pubfig_attributes.txt'
	URL_FILE = './Sample/dev_urls.txt'

	def __init__(self):
		if not os.path.exists(self.ATTRIBUTE_FILE):
			print('The attribute file [[ {} ]] is not in directory'.format(self.ATTRIBUTE_FILE))
			print('It will be downloaded...')
			response = requests.get('https://www.cs.columbia.edu/CAVE/databases/pubfig/download/pubfig_attributes.txt')
			with open(self.ATTRIBUTE_FILE, 'wb') as f:
				for data in tqdm(response.iter_content()):
					f.write(data)
			print('Downloaded successfully', end='\n\n')

		if not os.path.exists(self.URL_FILE):
			print('The attribute file [[ {} ]] is not in directory'.format(self.URL_FILE))
			print('It will be downloaded...')
			response = requests.get('https://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt')
			with open(self.URL_FILE, 'wb') as f:
				for data in tqdm(response.iter_content()):
					f.write(data)
			print('Downloaded successfully', end='\n\n')


	def download(self):
		# Read the files with pandas.DataFrame
		imgurl_df = pd.read_table(self.URL_FILE, sep='\t', skiprows=[0,1], names=['person', 'imagenum', 'url', 'rect', 'md5sum'])
		attr_df = pd.read_table(self.ATTRIBUTE_FILE, sep='\t', skiprows=[0,1], names=[
			'person', 'imagenum', 'Male', 'Asian', 'White', 'Black', 'Baby',
			'Child', 'Youth', 'Middle Aged', 'Senior', 'Black Hair', 'Blond Hair',
			'Brown Hair', 'Bald', 'No Eyewear', 'Eyeglasses', 'Sunglasses',
			'Mustache', 'Smiling', 'Frowning', 'Chubby', 'Blurry', 'Harsh Lighting',
			'Flash', 'Soft Lighting', 'Outdoor', 'Curly Hair', 'Wavy Hair',
			'Straight Hair', 'Receding Hairline', 'Bangs', 'Sideburns',
			'Fully Visible Forehead', 'Partially Visible Forehead',
			'Obstructed Forehead', 'Bushy Eyebrows', 'Arched Eyebrows',
			'Narrow Eyes', 'Eyes Open', 'Big Nose', 'Pointy Nose', 'Big Lips',
			'Mouth Closed', 'Mouth Slightly Open', 'Mouth Wide Open',
			'Teeth Not Visible', 'No Beard', 'Goatee', 'Round Jaw', 'Double Chin',
			'Wearing Hat', 'Oval Face', 'Square Face', 'Round Face', 'Color Photo',
			'Posed Photo', 'Attractive Man', 'Attractive Woman', 'Indian',
			'Gray Hair', 'Bags Under Eyes', 'Heavy Makeup', 'Rosy Cheeks',
			'Shiny Skin', 'Pale Skin', "5 o' Clock Shadow",
			'Strong Nose-Mouth Lines', 'Wearing Lipstick', 'Flushed Face',
			'High Cheekbones', 'Brown Eyes', 'Wearing Earrings', 'Wearing Necktie','Wearing Necklace'
			])

		# Merge two DataFrame using (person & imagenum)
		imgurl_df['key'] = imgurl_df['person'] + '_' + imgurl_df['imagenum'].astype(str)
		attr_df['key'] = attr_df['person'] + '_' + attr_df['imagenum'].astype(str)
		merged = imgurl_df.merge(attr_df, how='inner')

		# Before download the images,
		# make directory for saving the images
		if not os.path.isdir(self.ROOT):
			os.mkdir(self.ROOT)
		print('The pubfig images will be downloaded on [[ {} ]]'.format(self.ROOT))

		# Download images
		print('{} images downloaded from source urls'.format(len(merged)))
		print('Download start...')

		for iuk in tqdm(zip(merged.index, merged['url'], merged['key'])):
			idx, url, key = iuk
			fn = key + '.jpg'
			filepath = os.path.join(self.ROOT, fn)

			if os.path.exists(filepath):
				continue

			try:
				response = requests.get(url, timeout=2)
			except requests.exceptions.Timeout:
				# Timed out
				# The image link blocked or removed
				merged = merged.drop(index=idx)
				continue
			except:
				# Unknown URL
				# Remove the row on 'merged' DataFrame
				merged = merged.drop(index=idx)
				continue

			with open(filepath, 'wb') as img:
				for data in response.iter_content():
					img.write(data)
				response.close()
		print('All images {} download on {} successfully'.format(len(merged), self.ROOT))
		merged.to_csv('./Sample/pubfig_attr_merged.csv', index=False, encoding='utf-8')


	def to_dataset(self):
		img_files = glob('./Sample/pubfig/*')

		# Load the images
		img_keys = []
		img_list = []
		for ifn in tqdm(img_files):
			try:
				img = Image.open(ifn).resize((64,64))
			except:
				print(f'"{ifn}" cannot be resized by 64x64')
				continue
			img = np.asarray(img)
   
			if img.size == 12288:
				key = os.path.basename(ifn).replace('.jpg', '')
				img_list.append(img)
				img_keys.append(key)

		# Load the attribute file
		attribute = pd.read_csv('./Sample/pubfig_attr_merged.csv', encoding='utf-8')
		attribute = attribute[attribute['key'].isin(img_keys)]

		TARGET_NAME = 'Male'
		BIAS_NAME = 'Heavy Makeup'

		# Convert Target and Bias to categorical
		def categorize(score):
			if score > 0:
				return 1
			else:
				return 0

		vfunc = np.vectorize(categorize)

		target_vect = attribute[TARGET_NAME].to_numpy()
		target_vect = vfunc(target_vect)
		bias_vect = attribute[BIAS_NAME].to_numpy()
		bias_vect = vfunc(bias_vect)

		# Make images to DataFrame (for using aif360)
		temp = [im.ravel() for im in img_list]
		temp_df = pd.DataFrame(temp)

		# Add column
		temp_df[TARGET_NAME] = target_vect
		temp_df[BIAS_NAME] = bias_vect

		# Make dataset
		dataset = aifData(df=temp_df, label_name=TARGET_NAME,
			favorable_classes=[1],
			protected_attribute_names=[BIAS_NAME],
			privileged_classes=[[1]])

		result = {
			'aif_dataset': dataset,
			'image_list': img_list,
			'attribute': attribute,
			'target': target_vect,
			'bias': bias_vect
		}

		return result



if __name__ == "__main__": 
	ds = AdultDataset()
	#ds.download()
	dataset = ds.to_dataset()