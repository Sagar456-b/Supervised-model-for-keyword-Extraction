# Importing spaCy for natural language processing tasks
import spacy

# Importing NLTK libraries for natural language processing tasks such as tokenization and lemmatization
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

# Uncomment these lines to download necessary NLTK resources if not already installed
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

# Load the small English model from spaCy
nlp = spacy.load("en_core_web_sm")

# Import SSL library to handle HTTPS requests securely
import ssl

# Set up SSL context to bypass certificate verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # If the current SSL module does not support creating unverified contexts, do nothing
    pass
else:
    # If it does, use the unverified context by default for HTTPS requests
    ssl._create_default_https_context = _create_unverified_https_context

def get_wordnet_pos(treebank_tag):
    """
    Convert POS tags from the Penn Treebank format to a format compatible with WordNet lemmatization.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ  # Adjective
    elif treebank_tag.startswith('V'):
        return wordnet.VERB  # Verb
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN  # Noun
    elif treebank_tag.startswith('R'):
        return wordnet.ADV   # Adverb
    else:
        return wordnet.NOUN  # Default to noun if no match

def preprocess_text(text):
    """
    Preprocess the input text by removing named entities, converting to lowercase, tokenizing,
    removing stopwords and conjunctive adverbs, and lemmatizing the tokens.
    """
    # Set of conjunctive adverbs to remove from the text
    conjunctive_adverbs = {
        'accordingly', 'additionally', 'also', 'alternatively', 'anyway', 'besides', 'consequently',
        'conversely', 'elsewhere', 'equally', 'finally', 'further', 'furthermore', 'hence', 'however',
        'indeed', 'instead', 'likewise', 'meanwhile', 'moreover', 'namely', 'nevertheless', 'next',
        'nonetheless', 'otherwise', 'similarly', 'still', 'subsequently', 'then', 'thereafter', 'therefore',
        'thus', 'undoubtedly'
    }

    # Analyze the text using spaCy to remove named entities
    doc = nlp(text)
    text_without_entities = ' '.join([token.text for token in doc if not token.ent_type_])

    # Normalize the text, split into sentences, and initialize lemmatizer and stopwords
    cleaned_text = text_without_entities.lower().replace('\n', ' ').replace('\r', '').strip()
    sentences = sent_tokenize(cleaned_text)
    stop_words = set(stopwords.words('english')) | conjunctive_adverbs
    lemmatizer = WordNetLemmatizer()
    preprocessed_sentences = []

    # Process each sentence
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        filtered_words = [
            lemmatizer.lemmatize(word, get_wordnet_pos(pos))
            for word, pos in tagged_words
            if word.isalnum() and word.lower() not in stop_words
        ]
        preprocessed_sentences.append(' '.join(filtered_words))

    # Return the processed text
    return ' '.join(preprocessed_sentences)
