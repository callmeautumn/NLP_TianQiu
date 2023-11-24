# This code is functionally identical to the code in Week 6 classification notebook
# This has been separated into it's own pythons script for brevity

# Imports
import nltk
import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Downloads 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Function originally from: https://www.programcreek.com/python/?CodeExample=get%20wordnet%20pos
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Edit dataframe
df = pd.read_csv('../data/myers_briggs_comments.tsv', sep='\t')
df = df.drop('comment_id', axis=1)
df = df.drop('parent_comment_id', axis=1)

# Lemmatization
lemmatizer = WordNetLemmatizer()
for index, row in df.iterrows():
    comment = str(row['comment'])
    lemmitized_comment = " ".join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in comment.split() if word not in stopwords.words('english')])
    df.loc[index, 'comment'] = lemmitized_comment

# Create classes
le = LabelEncoder()
df['feat_to_classify'] = df['personality type'].str.strip().str[0]
df['class_label'] = le.fit_transform(df['feat_to_classify'])
class_names = list(le.classes_)

# Extract comments and class labels
comments = df["comment"].values.tolist()
class_labels = df["class_label"].values.tolist()

# Get test train split
X_train, X_test, y_train, y_test = train_test_split(comments, class_labels, test_size=0.3, random_state=42)

# Get train test split
df_train = pd.DataFrame(list(zip(X_train, y_train)))
df_train = df_train.dropna()
df_test = pd.DataFrame(list(zip(X_test, y_test)))
df_test = df_test.dropna()

# Save class names
cn_df = pd.DataFrame(class_names, columns=['class_names'])
cn_df.to_csv('../data/mb_class_labels.tsv', sep="\t", index=False)

# Save datasets
df_train.to_csv('../data/mb_processed_train.tsv', sep="\t", index=False)
df_test.to_csv('../data/mb_processed_test.tsv', sep="\t", index=False)
print('Dataset processed!')