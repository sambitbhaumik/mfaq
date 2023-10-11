from datasets import load_dataset
import tldextract
from laserembeddings import Laser
import pandas as pd
from collections import defaultdict
from transformers import BertTokenizerFast, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

hr_dataset = load_dataset("clips/mfaq", "hr")
en_dataset = load_dataset("clips/mfaq", "en")

def extract_root_domain(domain_url):
    return tldextract.extract(domain_url).domain

en_domains = list(map(extract_root_domain, en_dataset['train']['domain']))
en_domains = list(set(en_domains))  # to get unique domains
domain_datasets = {}

hr_domains = list(map(extract_root_domain, hr_dataset['train']['domain']))
hr_domains = list(set(hr_domains))  # to get unique domains

en_hr = [x for x in en_domains if x in hr_domains] # stores common root domains in both languages

# Filter the `en_dataset` and `hr_dataset` to only include the common domains

en_dataset_filtered = en_dataset.filter(lambda example: extract_root_domain(example['domain']) in en_hr)
hr_dataset_filtered = hr_dataset.filter(lambda example: extract_root_domain(example['domain']) in en_hr)

en_list = ['lottery.com', 'tripadvisor.in', 'esky.co.ke', 'tripadvisor.com.au', 'flowers4malmo.com', 'tripadvisor.cz', 'hotels.com', 'tripadvisor.fi', 'flowers4ibiza.com', 'plus500.com.au', 'hotels.ng', 'bosch-diy.com', 'flowers4zaragoza.com', 'flowers4brussels.com', 'eucerin.sg', 'flowers4latvia.com', 'hostinger.my', 'tripadvisor.co.hu', 'hostinger.dk', 'hostinger.com', 'rentalcargroup.com', 'globusbet.com', 'flowers4tenerife.com', 'flowers4armenia.com', 'flowers4dominicanrepublic.com', 'eucerin.ua', 'esky.com.sg', 'tripadvisor.co.za', 'plus500.co.nz', 'esky.com.my', 'iqos.com', 'hostinger.in', 'tripadvisor.co.kr', 'vpnmentor.com', 'flowers4belarus.com', 'tripadvisor.com.hk', 'letina.com', 'esky.co.uk', 'esky.eu', 'plus500.com.sg', 'esky.com.ng', 'poland-yacht-registration.com', 'flowers4london.com', 'radissonhotels.com', 'tripadvisor.com.my', 'flowers4spain.com', 'esky.ie', 'eucerin.my', 'safetydetective.com', 'esky.com.hk', 'sonuker.com', 'medorahotels.com', 'tripadvisor.se', 'tripadvisor.sk', 'uber.com', 'tripadvisor.com.sg', 'tripadvisor.co.uk', 'flowers4finland.com', 'mrinsta.com', 'plus500.co.uk', 'hostinger.co.uk', 'plus500.com', 'hotels.cn', 'flowers4valletta.com', 'eucerin.com', 'wizcase.com', 'rayhaber.com', 'johnnybet.com', 'tripadvisor.com.ve', 'plus500.co.za', 'websiteplanet.com', 'tripadvisor.com.tw', 'expediagroup.com', 'eucerin.co.za', 'toyota.co.uk', 'tripadvisor.com.vn', 'tripadvisor.com.ph', 'tripadvisor.com', 'toyota.com.cy', 'eucerin.ca', 'eucerin.co.uk', 'esky.com', 'flowers4milan.com', 'esky.com.eg', 'hostinger.jp', 'tripadvisor.ca', 'tripadvisor.co.nz', 'nikal.com.au', 'flowers4sanmarino.com', 'tripadvisor.ie', 'hostinger.se']
en_filtered = en_dataset_filtered.filter(lambda example: example['domain'] in en_list)

en_sentences = [pair["question"] for page in en_filtered for pair in page["qa_pairs"]]
hr_sentences = [pair["question"] for page in hr_dataset_filtered for pair in page["qa_pairs"]]
hr_answers = [pair["answer"] for page in hr_dataset_filtered for pair in page["qa_pairs"]]

laser = Laser()
en_embeddings = laser.embed_sentences(en_sentences, lang='en')
hr_embeddings = laser.embed_sentences(hr_sentences, lang='hr')

# Perform L2 normalization on the embeddings
en_embeddings = normalize(en_embeddings, norm='l2')
hr_embeddings = normalize(hr_embeddings, norm='l2')

# Compute cosine similarities between English and Croatian sentence embeddings
similarities = cosine_similarity(en_embeddings, hr_embeddings)

# Set the similarity threshold
similarity_threshold = 0.8

# Find the index of the most similar Croatian sentence for each English sentence
most_similar_indices = similarities.argmax(axis=1)
# Align English and Croatian sentences that pass the similarity threshold
aligned_sentences = []
for i in range(len(en_sentences)):
    similarity_score = similarities[i, most_similar_indices[i]]
    if similarity_score >= similarity_threshold:
        idx = hr_sentences.index(hr_sentences[most_similar_indices[i]])
        aligned_sentences.append((en_sentences[i], hr_sentences[most_similar_indices[i]], hr_answers[idx]))

df = pd.DataFrame(aligned_sentences, columns=['question','cr_question','answer'])
df = df.drop(1)
df.drop('cr_question', axis=1, inplace=True)

